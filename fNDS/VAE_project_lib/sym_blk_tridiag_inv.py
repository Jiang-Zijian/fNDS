"""
Pytorch implementation of functions to perform the inversion
of a symmetric block tridiagonal matrix. Ported from Evan Archer's
implementation here: https://github.com/earcher/vilds/blob/master/code/lib/sym_blk_tridiag_inv.py
"""

import torch
import numpy as np
from torch.autograd import Variable

def compute_sym_blk_tridiag(AA, BB, iia=None, iib=None):
    """
    Symbolically compute block tridiagonal terms of the inverse of a *symmetric* block tridiagonal matrix.

    All input & output assumed to be stacked torch tensors. Note that the function expects the off-diagonal
    blocks of the upper triangle & returns the lower-triangle (the transpose). Matrix is assumed symmetric so
    this doesn't really matter, but be careful.
    Input:
    AA - (T x d x d) diagonal blocks
    BB - (T-1 x d x d) off-diagonal blocks (upper triangle)
    iia - (T x 1) block index of AA for the diagonal
    iib - (T-1 x 1) block index of BB for the off-diagonal
    Output:
    D  - (T x d x d) diagonal blocks of the inverse
    OD - (T-1 x d x d) off-diagonal blocks of the inverse (lower triangle)
    S  - (T-1 x d x d) intermediary matrix computation used in inversion algorithm

    From:
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"
    Note: Could be generalized to non-symmetric matrices, but it is not currently implemented.
    (c) Evan Archer, 2015

    Implemented in Pytorch by JMP, 2017.
    """
    BB = -BB  # To match the convention of Jain et al.

    # number of blocks
    if iia is None:
        nT = AA.size(0)
    else:
        nT = iia.size(0)

    # dimension of block
    d = AA.size(1)

    if iia is None:
        iia = range(nT)
    if iib is None:
        iib = range(nT - 1)

    III = Variable(torch.eye(d), requires_grad=False)

    S = Variable(torch.Tensor(nT - 1, d, d))
    S[-1] = torch.mm(BB[iib[-1]], AA[iia[-1]].inverse())
    for i in range(nT - 3, -1, -1):
        S[i] = (torch.mm(BB[iib[i]], torch.inverse(AA[iia[i + 1]] -
             torch.mm(S[i + 1], BB[iib[i + 1]].t()))))

    D = Variable(torch.Tensor(nT, d, d))
    D[0] = (AA[iia[0]] - torch.mm(BB[iib[0]], S[0].t())).inverse()

    for i in range(1, nT - 1):
        Q = (AA[iia[i]] - torch.mm(BB[iib[i]], S[i].t())).inverse()
        D[i] = torch.mm(Q, III + torch.mm(BB[iib[i - 1]].t(),
                        torch.mm(D[i - 1], S[i - 1])))

    D[-1] = torch.mm(AA[iia[-1]].inverse(), III + torch.mm(BB[iib[-1]].t(),
                        torch.mm(D[-2], S[-1])))

    OD = Variable(torch.Tensor(nT - 1, d, d))
    for i in range(nT - 1):
        OD[i] = torch.mm(S[i].t(), D[i])

    return D, OD, S


def compute_sym_blk_tridiag_inv_b(S,D,b):
    """
    Symbolically solve Cx = b for x, where C is assumed to be *symmetric* block matrix.
    Input:
    D  - (T x d x d) diagonal blocks of the inverse
    S  - (T-1 x d x d) intermediary matrix computation returned by
         the function compute_sym_blk_tridiag
    Output:
    x - (T x d) solution of Cx = b
    From:
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"
    (c) Evan Archer, 2015

    Implemented in Pytorch by JMP, 2017.
    """
    nT, d = b.size()

    p = Variable(torch.Tensor(nT, d))
    p[-1] = b[-1]
    for i in range(nT - 2, -1, -1):
        p[i] = b[i] + torch.mv(S[i], p[i + 1])

    q = Variable(torch.Tensor(nT - 1, d))
    x = Variable(torch.Tensor(nT, d))
    q[0] = torch.mv(S[0].t(), torch.mv(D[0], b[0]))
    x[0] = torch.mv(D[0], p[0])
    for i in range(1, nT - 1):
        q[i] = torch.mv(S[i].t(), q[i - 1] + torch.mv(D[i], b[i]))
        x[i] = torch.mv(D[i], p[i]) + q[i - 1]
    x[-1] = torch.mv(D[-1], p[-1]) + q[-1]

    return x

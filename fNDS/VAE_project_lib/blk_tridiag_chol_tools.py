"""
Pytorch implementation of functions to perform the Cholesky factorization
of a block tridiagonal matrix. Ported from Evan Archer's implementation
here: https://github.com/earcher/vilds/blob/master/code/lib/blk_tridiag_chol_tools.py
"""

import torch
import numpy as np
from torch.autograd import Variable

def blk_tridiag_chol(A, B):
    """
    Compute the cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block
        off-diagonal matrix
    Outputs:
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky
    """
    R = [Variable(torch.Tensor(A.size())), Variable(torch.Tensor(B.size()))]
    L = torch.linalg.cholesky(A[0], upper=False)
    R[0][0] = L
    for i in range(B.size(0)):
#         C = torch.gesv(B[i], L)[0].t()
        C = torch.linalg.solve(L,B[i]).t()
        D = A[i + 1] - torch.matmul(C, C.t())
        L = torch.linalg.cholesky(D, upper=False)
        R[0][i + 1], R[1][i] = L, C

    return R

def blk_chol_inv(A, B, b, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix
    b - [T x n] tensor

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)
    Outputs:
    x - solution of Cx = b
    """
    X = Variable(torch.Tensor(A.size(0), A.size(1)))

    if transpose:
        A = torch.transpose(A, 1, 2)
        B = torch.transpose(B, 1, 2)

    if lower:
#         x = torch.gesv(b[0], A[0])[0]
        x = torch.linalg.solve(A[0],b[0])
        X[0] = x
        for i in range(B.size(0)):
#             x = torch.gesv(b[i + 1].unsqueeze(1) - torch.mm(B[i], x), A[i + 1])[0]
#             print("yes",B[i].shape,x.shape)
            x = torch.linalg.solve(A[i + 1],b[i + 1] - B[i]@x)
            X[i + 1] = x
    else:
#         x = torch.gesv(b[-1], A[-1])[0]
        x = torch.linalg.solve(A[-1],b[-1])
#         print(x)
        X[-1] = x
        for i in range(-1, -B.size(0) - 1, -1):
#             x = torch.gesv(b[i - 1].unsqueeze(1) - torch.mm(B[i], x), A[i - 1])[0]
#             print("no",B[i].shape,x.shape)
#             print((B[i]@x).shape)
#             print(b[i - 1].shape)
#             print(A[i - 1].shape,(b[i - 1] - B[i]@x).shape)
            x = torch.linalg.solve(A[i - 1],b[i - 1] - B[i]@x)
#             print(x.shape)
            X[i - 1] = x

    return X

def blk_chol_mtimes(A, B, x, lower = True, transpose = False):
    """
    Evaluate Cx = b, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)
    Outputs:
    b - result of Cx = b
    """
    b = Variable(torch.Tensor(A.size(0), A.size(1)))

    if transpose:
        A = torch.transpose(A, 1, 2)
        B = torch.transpose(B, 1, 2)

    if lower:
        b[0] = torch.mv(A[0], x[0])
        for i in range(B.size(0)):
            b[i + 1] = torch.mv(A[i + 1], x[i + 1]) + torch.mv(B[i], x[i])
    else:
        for i in range(B.size(0)):
            b[i] = torch.mv(A[i], x[i]) + torch.mv(B[i], x[i + 1])
        b[-1] = torch.mv(A[-1], x[-1])

    return b

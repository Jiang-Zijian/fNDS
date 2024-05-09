import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from importlib import reload
import math
from scipy.optimize import fsolve

def chol(cov):
    """
    cov = a@a.T
    return a
    """
    return torch.linalg.cholesky(cov)

from VAE_project_lib.sym_blk_tridiag_inv import *
from VAE_project_lib.blk_tridiag_chol_tools import *




class LDS(torch.nn.Module):
    """
    Generative model with linear dynamical system latent and Gaussian emission probability
    """
    def __init__(self,GenerativeParams):
        super(LDS, self).__init__()
        self.xdim = GenerativeParams['xdim']
        self.zdim = GenerativeParams['zdim']       
        self.mu1 = nn.Parameter(torch.Tensor(GenerativeParams['mu1']))
        self.Q1chol = nn.Parameter(torch.Tensor(GenerativeParams['Q1chol']))
        self.A = nn.Parameter(torch.Tensor(GenerativeParams['A']))
        self.Qchol = nn.Parameter(torch.Tensor(GenerativeParams['Qchol']))        
        self.add_module('NN', GenerativeParams['NN'])
        # The covariance of Gaussian emission probability
        self.Rchol = nn.Parameter(torch.Tensor(GenerativeParams['Rchol']))
    def zsample(self,T):
        """
        Sample z from the prior
        T (int): time bins to generate
        """       
        with torch.no_grad():
            norm_samp = torch.normal(torch.zeros(T, self.zdim))
            z_vals = torch.zeros([T, self.zdim])
            z_vals[0] = self.mu1 + self.Q1chol@norm_samp[0]
            for ii in range(T-1):
                z_vals[ii+1] = self.A@z_vals[ii] + self.Qchol@norm_samp[ii+1]
        return z_vals
    def xsample(self,T,returnz=False):
        """
        Sample x from the Generative model
        T (int): time bins to generate
        """ 
        with torch.no_grad():
            z_vals = self.zsample(T)
            norm_samp = torch.normal(torch.zeros(T, self.xdim))
            xsample = self.NN(z_vals)+norm_samp@torch.diag(torch.abs(self.Rchol))
            if returnz:
                return xsample,z_vals
            else:
                return xsample
    def logdensity(self,x,z,beta=1):
        """
        Calculate logP(x|z)+\beta*logP(z), given x and z, constants unrelated to model parameters are 
        omitted. 
        x (tensor): a sample from observation
        z (tensor): a sample from latent
        beta (float): the beta in beta VAE
        """
        self.Q = self.Qchol@(self.Qchol.T)
        self.Q1 = self.Q1chol@(self.Q1chol.T)        
        self.Lambda = torch.inverse(self.Q)
        self.Lambda1 = torch.inverse(self.Q1)
        self.Rinv = 1.0/(self.Rchol**2)
        xpred = self.NN(z)
        resx = x-xpred
        T = x.shape[0]
        resz = z[1:]-z[:-1]@(self.A.T)
        resz0 = z[0]-self.mu1
        loss = -beta*0.5*torch.sum((resz.T@resz)*self.Lambda)
        loss+= -beta*0.5*(resz0@self.Lambda1@resz0)
        loss+= -beta*0.5*(T-1)*torch.sum(torch.log(torch.diag(self.Q)))
        loss+= -beta*0.5*torch.sum(torch.log(torch.diag(self.Q1)))
        # Special for Gaussian emission probability
        loss+= -0.5*torch.sum((resx.T@resx)*torch.diag(self.Rinv))
        loss+= 0.5*T*torch.sum(torch.log(self.Rinv))
        return loss
    def decode(self,z):
        """
        Sample an observation according to P(x|z)
        z (tensor): a sample from latent
        """
        with torch.no_grad():
            T = z.shape[0]
            norm_samp = torch.normal(torch.zeros(T, self.xdim))
            xsample = self.NN(z)+norm_samp@torch.diag(torch.abs(self.Rchol))
            return xsample
        
class pLDS(LDS):
    """
    Generative model with linear dynamical system latent and Poisson emission probability
    """
    def __init__(self,GenerativeParams):
        nn.Module.__init__(self)
        self.xdim = GenerativeParams['xdim']
        self.zdim = GenerativeParams['zdim']
        self.mu1 = Parameter(torch.Tensor(GenerativeParams['mu1']))
        self.Q1chol = Parameter(torch.Tensor(GenerativeParams['Q1chol']))
        self.A = Parameter(torch.Tensor(GenerativeParams['A']))
        self.Qchol = Parameter(torch.Tensor(GenerativeParams['Qchol']))      
        self.add_module('NN', GenerativeParams['NN'])
        
    def xsample(self,T,returnz=False):
        """
        Sample x from the Generative model
        T (int): time bins to generate
        """ 
        with torch.no_grad():
            z_vals = self.zsample(T)
            norm_samp = torch.normal(torch.zeros(T, self.xdim))
            xsample = torch.poisson(self.NN(z_vals))
            if returnz:
                return xsample,z_vals
            else:
                return xsample
    def logdensity(self,x,z,beta=1):
        """
        Calculate logP(x|z)+\beta*logP(z), given x and z. Constants unrelated to model parameters are 
        omitted. 
        x (tensor): a sample from observation
        z (tensor): a sample from latent
        beta (float): the beta in beta VAE
        """
        self.Q = self.Qchol@(self.Qchol.T)
        self.Q1 = self.Q1chol@(self.Q1chol.T)        
        self.Lambda = torch.inverse(self.Q)
        self.Lambda1 = torch.inverse(self.Q1)
        xpred = self.NN(z)
        resx = x-xpred
        T = x.shape[0]
        resz = z[1:]-z[:-1]@(self.A.T)
        resz0 = z[0]-self.mu1
        loss = -beta*0.5*torch.sum((resz.T@resz)*self.Lambda)
        loss+= -beta*0.5*(resz0@self.Lambda1@resz0)
        loss+= -beta*0.5*(T-1)*torch.sum(torch.log(torch.diag(self.Q)))
        loss+= -beta*0.5*torch.sum(torch.log(torch.diag(self.Q1)))
        # Special for Poisson emission probability
        loss+= torch.sum(x * torch.log(xpred)  - xpred - torch.special.gammaln(x + 1.0))
        return loss
    def decode(self,z):
        """
        Sample an observation according to P(x|z)
        z (tensor): a sample from latent
        """
        with torch.no_grad():
            T = z.shape[0]
            xsample = torch.poisson(self.NN(z))
            return xsample 

class nLDS(torch.nn.Module):
    """
    Generative model with non-linear latent dynamics and Gaussian emission probability
    """
    def __init__(self,GenerativeParams):
        super(nLDS, self).__init__()
        self.xdim = GenerativeParams['xdim']
        self.zdim = GenerativeParams['zdim']       
        self.mu1 = nn.Parameter(torch.Tensor(GenerativeParams['mu1']))
        self.Q1chol = nn.Parameter(torch.Tensor(GenerativeParams['Q1chol']))
        self.Qchol = nn.Parameter(torch.Tensor(GenerativeParams['Qchol']))        
        self.add_module('NN', GenerativeParams['NN'])
        self.Rchol = nn.Parameter(torch.Tensor(GenerativeParams['Rchol']))
        # the neural network for parameterize non-linear latent dynamics
        self.add_module('nN', GenerativeParams['nN'])
    def zsample(self,T):
        """
        Sample z from the prior P(z)
        T (int): time bins to generate
        """
        with torch.no_grad():
            norm_samp = torch.normal(torch.zeros(T, self.zdim))
            z_vals = torch.zeros([T, self.zdim])
            z_vals[0] = self.mu1 + self.Q1chol@norm_samp[0]
            for ii in range(T-1):
                z_vals[ii+1] = self.nN(z_vals[ii].view(1,-1)).view(-1) + self.Qchol@norm_samp[ii+1]
        return z_vals
    def xsample(self,T,returnz=False):
        """
        Sample x from the Generative model P(x,z)
        T (int): time bins to generate
        """ 
        with torch.no_grad():
            z_vals = self.zsample(T)
            norm_samp = torch.normal(torch.zeros(T, self.xdim))
            xsample = self.NN(z_vals)+norm_samp@torch.diag(torch.abs(self.Rchol))
            if returnz:
                return xsample,z_vals
            else:
                return xsample
    def logdensity(self,x,z,beta=1):
        """
        Calculate logP(x|z)+\beta*logP(z), given x and z. Constants unrelated to model parameters are 
        omitted. 
        x (tensor): a sample from observation
        z (tensor): a sample from latent
        beta (float): the beta in beta VAE
        """
        self.Q = self.Qchol@(self.Qchol.T)
        self.Q1 = self.Q1chol@(self.Q1chol.T)        
        self.Lambda = torch.inverse(self.Q)
        self.Lambda1 = torch.inverse(self.Q1)
        self.Rinv = 1.0/(self.Rchol**2)
        xpred = self.NN(z)
        resx = x-xpred
        T = x.shape[0]
        resz = z[1:]-self.nN(z[:-1]) # Special for nLDS       
        resz0 = z[0]-self.mu1
        loss = -beta*0.5*torch.sum((resz.T@resz)*self.Lambda)
        loss+= -beta*0.5*(resz0@self.Lambda1@resz0)
        loss+= -beta*0.5*(T-1)*torch.sum(torch.log(torch.diag(self.Q)))
        loss+= -beta*0.5*torch.sum(torch.log(torch.diag(self.Q1)))
        loss+= -0.5*torch.sum((resx.T@resx)*torch.diag(self.Rinv))
        loss+= 0.5*T*torch.sum(torch.log(self.Rinv))
        return loss
    def decode(self,z):
        """
        Sample an observation according to P(x|z)
        z (tensor): a sample from latent
        """
        with torch.no_grad():
            T = z.shape[0]
            norm_samp = torch.normal(torch.zeros(T, self.xdim))
            xsample = self.NN(z)+norm_samp@torch.diag(torch.abs(self.Rchol))
            return xsample
        
class inverse_LDS(torch.nn.Module):
    """
    Approximate variational posterior(Recognition model) with tri-diagonal block structure
    """
    def __init__(self,RecognitionParams):
        super(inverse_LDS,self).__init__()
        self.xdim = RecognitionParams["xdim"]
        self.zdim = RecognitionParams["zdim"]
        self.add_module('mu', RecognitionParams['NN_Mu'])
        self.add_module('Lambda', RecognitionParams['NN_Lambda'])
        self.add_module('LambdaX', RecognitionParams['NN_LambdaX'])
    def initialize_posterior_distribution(self,data,beta=1.0):
        """
        Calculate \beta*H(q(z|x)), where H(q(z|x)) represents the entropy of the posterior
        data (tensor): a sample from observation, with shape T*xdim
        beta (float): the beta in beta-VAE
        """
        self.Tt = data.shape[0]
        self.postX = self.mu(data)
        self.AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False) # 3 is ensure to make AAChol is full-ranked under the Lambda structure in this experiment, can be increased based on actual needs. 
        self.BBChol = self.LambdaX(data).view(-1, self.zdim, self.zdim)
        diagsquare = torch.bmm(self.AAChol, self.AAChol.transpose(1 ,2))
        odsquare = torch.bmm(self.BBChol, self.BBChol.transpose(1, 2))
        self.AA = diagsquare + torch.cat([Variable(torch.zeros(1, self.zdim, self.zdim), requires_grad=False), odsquare]) + 1e-6 * Variable(torch.eye(self.zdim), requires_grad=False)# to ensure the positive define under the default numerical precision of PyTorch
        self.BB = torch.bmm(self.AAChol[:-1], self.BBChol.transpose(1, 2))
        self.the_chol = blk_tridiag_chol(self.AA, self.BB)
        self.logdet = 0
        for i in range(self.the_chol[0].size(0)):
            self.logdet += -2 * self.the_chol[0][i].diag().log().sum()
        self.entropy = beta*0.5*self.logdet
        return self.entropy
    def getSample(self):
        """
        Obtain a single sample from q(z|x), requires gradient
        """
        normSamps = torch.randn([self.Tt, self.zdim],requires_grad=False)
        return self.postX + blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)
    def encode(self,data):
        """
        Transfer the data into latent dynamics by sampling from q(z|x)
        """
        with torch.no_grad():
            Tt = data.shape[0]
            postX = self.mu(data)
            AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False)
            BBChol = self.LambdaX(data).view(-1, self.zdim, self.zdim)
            diagsquare = torch.bmm(AAChol, AAChol.transpose(1 ,2))
            odsquare = torch.bmm(BBChol, BBChol.transpose(1, 2))
            AA = diagsquare + torch.cat([Variable(torch.zeros(1, self.zdim, self.zdim), requires_grad=False), odsquare]) + 1e-6 * Variable(torch.eye(self.zdim), requires_grad=False)
            BB = torch.bmm(AAChol[:-1], BBChol.transpose(1, 2))
            the_chol = blk_tridiag_chol(AA, BB)
            normSamps = torch.randn([Tt, self.zdim],requires_grad=False)
            return postX + blk_chol_inv(the_chol[0], the_chol[1], normSamps, lower=False, transpose=True)

        
class mf_inverse_LDS(torch.nn.Module):
    '''
    Mean-field variational posterior, i.e. the posterior has a block diagonal structure. 
    '''

    def __init__(self,RecognitionParams):
        super(mf_inverse_LDS, self).__init__()
        self.xdim = RecognitionParams["xdim"]
        self.zdim = RecognitionParams["zdim"]
        self.add_module('mu', RecognitionParams['NN_Mu'])
        self.add_module('Lambda', RecognitionParams['NN_Lambda'])
    def initialize_posterior_distribution(self,data,beta=1.0):
        self.Tt = data.shape[0]
        self.postX = self.mu(data)
        self.AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False)
        self.AA = torch.bmm(self.AAChol, self.AAChol.transpose(1 ,2))
        self.logdet = 0
        for i in range(self.Tt):
            self.logdet += 1 * torch.log(torch.linalg.det(self.AA[i]))
        self.entropy = beta*0.5*self.logdet
        return self.entropy

    def getSample(self):
        normSamps = torch.randn([self.Tt, self.zdim,1],requires_grad=False)
        retSamp = torch.mean(torch.bmm(self.AAChol, normSamps),axis=2)
        return retSamp+self.postX
    def encode(self,data):
        with torch.no_grad():
            Tt = data.shape[0]
            postX = self.mu(data)
            AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False)
            normSamps = torch.randn([Tt, self.zdim,1],requires_grad=False)
            return postX+torch.squeeze(torch.bmm(AAChol, normSamps))

class inverse_LDS_test(torch.nn.Module):
    """
    inverse LDS can be trained by ground true z, for debug usage
    
    """
    def __init__(self,RecognitionParams,GenerativeParams):
        super(inverse_LDS_test,self).__init__()
        self.xdim = RecognitionParams["xdim"]
        self.zdim = RecognitionParams["zdim"]
        self.add_module('mu', RecognitionParams['NN_Mu'])
        self.add_module('Lambda', RecognitionParams['NN_Lambda'])
        self.add_module('LambdaX', RecognitionParams['NN_LambdaX'])
        # none-parameter
#         self.mu1 = torch.Tensor(GenerativeParams['mu1']).detach()
        self.register_buffer("mu1",GenerativeParams['mu1'])
#         self.Q1chol = torch.Tensor(GenerativeParams['Q1chol']).detach()
        self.register_buffer("Q1chol",GenerativeParams['Q1chol'])
#         self.A = torch.Tensor(GenerativeParams['A']).detach()
        self.register_buffer("A",GenerativeParams['A'])
#         self.Qchol = torch.Tensor(GenerativeParams['Qchol']).detach()
        self.register_buffer("Qchol",GenerativeParams['Qchol'])
        self.register_buffer("Q",self.Qchol@(self.Qchol.T))
        self.register_buffer("Q1",self.Q1chol@(self.Q1chol.T))
        self.register_buffer("LLambda",torch.inverse(self.Q))
        self.register_buffer("LLambda1",torch.inverse(self.Q1))
#         self.Q = self.Qchol@(self.Qchol.T)
#         self.Q1 = self.Q1chol@(self.Q1chol.T)        
#         self.Lambda = torch.inverse(self.Q)
#         self.Lambda1 = torch.inverse(self.Q1)
    def initialize_posterior_distribution(self,data,beta=1.0):
        self.Tt = data.shape[0]
        self.postX = self.mu(data)
        self.AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False)
        self.BBChol = self.LambdaX(data).view(-1, self.zdim, self.zdim)
        diagsquare = torch.bmm(self.AAChol, self.AAChol.transpose(1 ,2))
        odsquare = torch.bmm(self.BBChol, self.BBChol.transpose(1, 2))
        self.AA = diagsquare + torch.cat([Variable(torch.zeros(1, self.zdim, self.zdim), requires_grad=False), odsquare]) + 1e-6 * Variable(torch.eye(self.zdim), requires_grad=False)
        self.BB = torch.bmm(self.AAChol[:-1], self.BBChol.transpose(1, 2))
        self.the_chol = blk_tridiag_chol(self.AA, self.BB)
    # compute_sym_blk_tridiag 有内存泄露！！！
#         self.V, self.VV, self.S = compute_sym_blk_tridiag(self.AA, self.BB)
        self.logdet = 0
        for i in range(self.the_chol[0].size(0)):
            self.logdet += -2 * self.the_chol[0][i].diag().log().sum()
        self.entropy = beta*0.5*self.logdet
        sample=self.getSample()
        
        # calculate the p(z)
        resz = sample[1:]-sample[:-1]@(self.A.T)
        resz0 = sample[0]-self.mu1
        loss = -beta*0.5*torch.sum((resz.T@resz)*self.LLambda)
        loss+= -beta*0.5*(resz0@self.LLambda1@resz0)
        loss+= -beta*0.5*(T-1)*torch.sum(torch.log(torch.diag(self.Q)))
        loss+= -beta*0.5*torch.sum(torch.log(torch.diag(self.Q1)))        
        return self.entropy+loss
    def encode_loss(self,data,realz):
        self.Tt = data.shape[0]
        self.postX = self.mu(data)
        self.AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False)
        self.BBChol = self.LambdaX(data).view(-1, self.zdim, self.zdim)
        diagsquare = torch.bmm(self.AAChol, self.AAChol.transpose(1 ,2))
        odsquare = torch.bmm(self.BBChol, self.BBChol.transpose(1, 2))
        self.AA = diagsquare + torch.cat([Variable(torch.zeros(1, self.zdim, self.zdim), requires_grad=False), odsquare]) + 1e-6 * Variable(torch.eye(self.zdim), requires_grad=False)
        self.BB = torch.bmm(self.AAChol[:-1], self.BBChol.transpose(1, 2))
        self.the_chol = blk_tridiag_chol(self.AA, self.BB)
        normSamps = torch.randn([self.Tt, self.zdim],requires_grad=False)
        sample=self.postX + blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)
        return -torch.sum((sample-realz)**2)/self.Tt
    def getSample(self):
        normSamps = torch.randn([self.Tt, self.zdim],requires_grad=False)
        return self.postX + blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)
    def encode(self,data):
        with torch.no_grad():
            Tt = data.shape[0]
            postX = self.mu(data)
            AAChol = self.Lambda(data).view(-1, self.zdim, self.zdim) + Variable(3*torch.eye(self.zdim), requires_grad=False)
            BBChol = self.LambdaX(data).view(-1, self.zdim, self.zdim)
            diagsquare = torch.bmm(AAChol, AAChol.transpose(1 ,2))
            odsquare = torch.bmm(BBChol, BBChol.transpose(1, 2))
            AA = diagsquare + torch.cat([Variable(torch.zeros(1, self.zdim, self.zdim), requires_grad=False), odsquare]) + 1e-6 * Variable(torch.eye(self.zdim), requires_grad=False)
            BB = torch.bmm(AAChol[:-1], BBChol.transpose(1, 2))
            the_chol = blk_tridiag_chol(AA, BB)
            normSamps = torch.randn([Tt, self.zdim],requires_grad=False)
            return postX + blk_chol_inv(the_chol[0], the_chol[1], normSamps, lower=False, transpose=True)
        
####################################################################################################
                
class ge_model(torch.nn.Module):
    def __init__(self,GenerativeParams):
        super(ge_model,self).__init__()
        self.GenerativeParams = GenerativeParams
        self.Generative = LDS(self.GenerativeParams)
    def forward(self,dataxz):
        datax,dataz = dataxz
        return -1.0*self.Generative.logdensity(datax,dataz)
    
class re_model(torch.nn.Module):
    def __init__(self,RecognitionParams):
        super(re_model,self).__init__()
        self.RecognitionParams = RecognitionParams
        self.re = inverse_LDS(self.RecognitionParams)
    def forward(self,datax):
        return -1.0*self.re.initialize_posterior_distribution(datax)
    
class re_model_test(torch.nn.Module):
    def __init__(self,RecognitionParams,GenerativeParams,REC,re_loss=False):
        super(re_model_test,self).__init__()
        self.RecognitionParams = RecognitionParams
        self.GenerativeParams = GenerativeParams
        self.re = REC(self.RecognitionParams,self.GenerativeParams)
        self.re_loss = re_loss
    def forward(self,dataxz):
        datax,dataz = dataxz
        if self.re_loss:           
            return -1.0*self.re.encode_loss(datax,dataz)
        else:
            return -1.0*self.re.initialize_posterior_distribution(datax)        
        
class VAE(torch.nn.Module):
    def __init__(self,GenerativeParams,RecognitionParams,GEN,REC,beta=1.0):
        super(VAE,self).__init__()
        self.GenerativeParams = GenerativeParams
        self.RecognitionParams = RecognitionParams    
        self.GEN = GEN
        self.REC = REC
        self.Generative = self.GEN(self.GenerativeParams)
        self.Recognition = self.REC(self.RecognitionParams)
        self.beta=beta
    def forward(self,x):        
        entropy =self.Recognition.initialize_posterior_distribution(x,beta=self.beta)
        sample = self.Recognition.getSample()
        logdensity = self.Generative.logdensity(x,sample,beta=self.beta)
        return -1.0*(entropy+logdensity)

    
import random
def train_vae(model, datas, optimizer, device="cpu", epochs=10,bothxz=False):
    # 将模型发送到指定设备
    model.to(device)    
    # 训练模式
    model.train()
    if not bothxz:
        len_datas = len(datas)
    else:
        len_datas = len(datas[0])
    feed_list = np.arange(len_datas)
    for epoch in range(epochs):        
        total_loss = 0.0
        random.shuffle(feed_list)
        for i in feed_list:
            if not bothxz:
                data = datas[i].to(device)
            else:
                data = (datas[0][i].to(device),datas[1][i].to(device))            
            loss = model(data)
            # 后向传播和优化
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward(retain_graph=True)      # 计算梯度
            optimizer.step()       # 更新参数            
            # 累加损失
            total_loss += loss.item()
        # 计算平均损失
        avg_loss = total_loss / len_datas
        # 打印训练状态
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()
    def forward(self, x):
        return torch.exp(x)
        
class NN_2l(torch.nn.Module):
    def __init__(self,input_dim,middle_dim,output_dim,offdiag=False):
        super(NN_2l,self).__init__()
        self.layer1 = nn.Linear(input_dim,middle_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(middle_dim,output_dim)
        self.offdiag=offdiag
    def forward(self,x):
        if self.offdiag:
            x = torch.cat([x[:-1],x[1:]],dim=1)
        x = self.layer1(x)
#         x = torch.tanh(x)
        x = self.relu(x)
        x = self.layer2(x)
#         x = torch.tanh(x)
        return x 

class NN_5l(torch.nn.Module):
    def __init__(self,input_dim,middle_dims,output_dim,offdiag=False):
        super(NN_5l,self).__init__()
        self.layer1 = nn.Linear(input_dim,middle_dims[0])
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(middle_dims[0],middle_dims[1])
        self.layer3 = nn.Linear(middle_dims[1],middle_dims[2])
        self.layer4 = nn.Linear(middle_dims[3],middle_dims[4])
        self.layer5 = nn.Linear(middle_dims[4],output_dim)
        self.offdiag=offdiag
    def forward(self,x):
        if self.offdiag:
            x = torch.cat([x[:-1],x[1:]],dim=1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        return x 
class NN_5l_new(torch.nn.Module):
    def __init__(self,input_dim,middle_dims,output_dim,offdiag=False,diag_drift = False):
        super(NN_5l_new,self).__init__()
        self.layer1 = nn.Linear(input_dim,middle_dims[0])
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(middle_dims[0],middle_dims[1])
        self.layer3 = nn.Linear(middle_dims[1],middle_dims[2])
        self.layer4 = nn.Linear(middle_dims[3],middle_dims[4])
        self.layer5 = nn.Linear(middle_dims[4],output_dim)
        self.offdiag=offdiag
        self.diag_drift=diag_drift
    def forward(self,x):
        if self.diag_drift:
            x = torch.cat([torch.zeros(1,x.shape[1]),x,torch.zeros(1,x.shape[1])],dim=0)
            x = torch.cat([x[:-2],x[1:-1],x[2:]],dim=1)
        if self.offdiag:
            x = torch.cat([x[:-1],x[1:]],dim=1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        return x 

def zero_mean_initialize(linear_layer,data):
    # intialize the linear layer inorderto make the mean ouput for the give dataset is zero
    W0 = linear_layer.weight.detach()
    B0 = linear_layer.bias.detach()
    W0_new = (W0.T/(data@W0.T+B0).std(axis=0)).T
    B0_new = -(data@W0_new.T).mean(axis=0)
    linear_layer.weight.data = W0_new
    linear_layer.bias.data = B0_new

def z_projection_single(zv,zt):
    zv = torch.cat([zv,torch.ones([zv.shape[0],1])],axis=1)
    return torch.linalg.inv((zv.T@zv))@(zv.T@zt)
def proj(vaez,prj_m):
    vaez = torch.cat([vaez,torch.ones([vaez.shape[0],1])],axis=1)
    return vaez@prj_m 

def z_projection(recognition_model,datasx,datasz):
    if len(datasx.shape)==3:
        datasz_stacked = []
        for i in range(len(datasz)):
            datasz_stacked.append(datasz[i])
        datasz_stacked = torch.cat(datasz_stacked,axis=0)
        vaez_stacked = []
        for i in range(len(datasx)):
            vaez_stacked.append(myvae.Recognition.encode(x).detach())
        vaez_stacked = torch.cat(vaez_stacked,axis=0)
        vaez_stacked = torch.cat([vaez_stacked,torch.ones([vaez_stacked.shape[0],1])],axis=1)
    if len(datasx.shape)==2:
        datasz_stacked =datasz
        vaez = myvae.Recognition.encode(datasx).detach()
        vaez_stacked = torch.cat([vaez,torch.ones([vaez.shape[0],1])],axis=1)
    return torch.linalg.inv((vaez_stacked.T@vaez_stacked))@(vaez_stacked.T@datasz_stacked)

class dynamics(nn.Module):
    """
    general 2D 1st-order dynamics system
    """
    def __init__(self,dx_dt,dy_dt,dt=0.05):
        super(dynamics, self).__init__()
        self.dt = dt
        self.dx_dt = dx_dt
        self.dy_dt = dy_dt
    def forward(self,xy):
        x = xy[:,0].view(1,-1)
        y = xy[:,1].view(1,-1)
        x1 = self.dt*self.dx_dt(x,y)+x
        y1 = self.dt*self.dy_dt(x,y)+y
        return torch.cat([x1,y1],dim=1)
def trajectory(x0,mapping,T):
    x0 = torch.tensor(x0).view(1,-1)
    traj = [x0,]
    for i in range(T-1):
        traj.append(mapping(traj[i]))
    return torch.cat(traj,dim=0)

def trajectory_network(x0,deep,T,dt=0.05):
    x0 = torch.tensor(x0).view(1,-1).float()
    traj = [x0,]
    with torch.no_grad():
        for i in range(T-1):
            traj.append(traj[i]+dt*deep(traj[i]))
    return torch.cat(traj,dim=0)

class deepnet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(deepnet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # 或者其它激活函数，如nn.Sigmoid()、nn.Tanh()
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())  # 或者其它激活函数
        layers.append(nn.Linear(hidden_sizes[-1], output_size))        
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class dynamics_student(nn.Module):
    """
    general 2D 1st-order dynamics system
    """
    def __init__(self,NN,dt=0.05):
        super(dynamics_student, self).__init__()
        self.dt = dt
        self.add_module('dxy', NN)
    def forward(self,xy):
        return xy+self.dt*self.dxy(xy)
    
def find_stable_points(dx_dt, dy_dt):
    def equations(xy):
        x, y = xy
        return [dx_dt(x, y), dy_dt(x, y)]
   
    initial_guesses = [[1, 1], [-1, -1], [1, -1], [-1, 1]] 
    

    stable_points = []
    for guess in initial_guesses:
        solution = fsolve(equations, guess)
        if np.all(np.isclose(equations(solution), 0)):  
            stable_points.append(solution)
    
    return stable_points

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from entmax import sparsemax
from functools import partial
from torch import Tensor

from core.model.basis_functions import GaussianBasisFunctions
import math 
from core.model.tv2d_numba import prox_tv2d

from torch.autograd import Function

from numba import jit

@jit(nopython=True)
def isin(x, l):
    for i in l:
        if x==i:
            return True
    return False

@jit(nopython=True)        
def back(Y, dX, dY):
    neigbhours=list([(1,1)])
    del neigbhours[-1] 
    group=[(0,0)]
    del group[-1]
    n=0
    idx_grouped = [(200,200)for x in range(196)]
    count=0
    value=0
    s=0
    while True:
        if len(neigbhours)!=0:
            while len(neigbhours)!=0:
                if Y[neigbhours[0][0],neigbhours[0][1]] == value:
                    a = neigbhours[0][0]
                    b = neigbhours[0][1]
                    del neigbhours[0]
                    count+=1
                    s+=dY[a,b]
                    group.append((a,b))
                    idx_grouped[n]=(a,b)
                    n+=1
                    if b<13 and isin((a,b+1), idx_grouped)==False and isin((a,b+1), neigbhours)==False:
                        neigbhours.append((a,b+1))
                    if a<13 and isin((a+1,b), idx_grouped)==False and isin((a+1,b), neigbhours)==False:
                        neigbhours.append((a+1,b)) 
                    if b>0 and isin((a,b-1), idx_grouped)==False and isin((a,b-1), neigbhours)==False:
                        neigbhours.append((a,b-1)) 
                    if a>0 and isin((a-1,b), idx_grouped)==False and isin((a-1,b), neigbhours)==False:
                        neigbhours.append((a-1,b)) 
                else:
                    del neigbhours[0]
        else:
            if len(group)>0:
                o=s/count
                count=0
                for x in group:
                    dX[x[0],x[1]]=o
                group=[(0,0)]
                del group[0]
            
            if n>=196:
                break
            B=False
            for i in range(14):
                for j in range(14):
                    if isin((i,j), idx_grouped)==False:
                        value = Y[i,j]
                        s = dY[i,j]
                        count+=1
                        group.append((i, j))
                        idx_grouped[n] = (i, j)
                        n+=1
                        if j<13 and isin((i,j+1), idx_grouped)==False and isin((i,j+1), neigbhours)==False:
                            neigbhours.append((i,j+1))
                        if i<13 and isin((i+1,j), idx_grouped)==False and isin((i+1,j), neigbhours)==False:
                            neigbhours.append((i+1,j)) 
                        if j>0 and isin((i,j-1), idx_grouped)==False and isin((i,j-1), neigbhours)==False:
                            neigbhours.append((i,j-1)) 
                        if i>0 and isin((i-1,j), idx_grouped)==False and isin((i-1,j), neigbhours)==False:
                            neigbhours.append((i-1,j)) 
                        B=True
                        break
                if B:
                    break
    return dX

class TV2DFunction(Function):

    @staticmethod
    def forward(ctx, X, alpha=0.01, max_iter=35, tol=1e-2):
        torch.set_num_threads(8)
        ctx.digits_tol = int(-np.log10(tol)) // 2

        X_np = X.detach().cpu().numpy()
        n_rows, n_cols = X_np.shape
        Y_np = prox_tv2d(X_np.ravel(),
                         step_size=alpha / 2,
                         n_rows=n_rows,
                         n_cols=n_cols,
                         max_iter=max_iter,
                         tol=tol)
        
        
        Y_np = Y_np.reshape(n_rows, n_cols)
        Y = torch.from_numpy(Y_np)  # double-precision
        Y = torch.as_tensor(Y, dtype=X.dtype, device=X.device)
        ctx.save_for_backward(Y.detach())  # TODO figure out why detach everywhere

        return Y

    @staticmethod
    def backward(ctx, dY):
        #with torch.autograd.profiler.profile(use_cuda=True) as prof)
        torch.set_num_threads(8)
        Y, = ctx.saved_tensors
        """
        tic = perf_counter()
        dY_np = dY.cpu().numpy()
        dX_np = np.zeros((8,8))

        Y_np_round = Y.cpu().numpy().round(ctx.digits_tol)
        # TODO speed me up. Maybe with scikit-image label?
        uniq, inv = np.unique(Y_np_round, return_inverse=True)
        
        inv = inv.reshape((8,8))
        
        for j in range(len(uniq)):
            objs, n_objs = label(inv == j)
            for k in range(1, n_objs + 1):
                obj = objs == k
                obj_mean = (obj * dY_np).sum() / obj.sum()
                dX_np += obj_mean * obj
        #tac=perf_counter()
        #print(torch.as_tensor(dX_np, dtype=dY.dtype, device=dY.device))
        #print('vlad', tac-tic)
        #tic=perf_counter()
        """
        Y_np = np.array(Y.cpu()).round(ctx.digits_tol)
        dY_np = np.array(dY.cpu())
        dX = np.zeros((14,14))
        dX = back(Y_np, dX, dY_np)
        dX = torch.as_tensor(dX, dtype=dY.dtype, device=dY.device)
        #tac=perf_counter()
        #print(dX)
        #print('pedro', tac-tic)
        
        return dX, None 



class ContinuousSoftmaxFunction(torch.autograd.Function):

    @classmethod
    def _expectation_phi_psi(cls, ctx, Mu, Sigma):
        """Compute expectation of phi(t) * psi(t).T under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros((Mu.shape[0], 6, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            V[:, 0, start:offsets[j]]=basis_functions.integrate_t_times_psi_gaussian(Mu,Sigma).squeeze(-1)[:,:,0]
            V[:, 1, start:offsets[j]]=basis_functions.integrate_t_times_psi_gaussian(Mu,Sigma).squeeze(-1)[:,:,1]
            V[:, 2, start:offsets[j]]=basis_functions.integrate_t2_times_psi_gaussian(Mu,Sigma)[:,:,0,0]
            V[:, 3, start:offsets[j]]=basis_functions.integrate_t2_times_psi_gaussian(Mu,Sigma)[:,:,0,1]
            V[:, 4, start:offsets[j]]=basis_functions.integrate_t2_times_psi_gaussian(Mu,Sigma)[:,:,1,0]
            V[:, 5, start:offsets[j]]=basis_functions.integrate_t2_times_psi_gaussian(Mu,Sigma)[:,:,1,1]
            start = offsets[j]
        return V # [batch,6,N]


    @classmethod
    def _expectation_psi(cls, ctx, Mu, Sigma):
        """Compute expectation of psi under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        r = torch.zeros(Mu.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            r[:, start:offsets[j]] = basis_functions.integrate_psi_gaussian(Mu, Sigma).squeeze(-2).squeeze(-1)
            start = offsets[j]
        return r # [batch,N]

    @classmethod
    def _expectation_phi(cls, ctx, Mu, Sigma):
        v = torch.zeros(Mu.shape[0], 6, dtype=ctx.dtype, device=ctx.device)
        v[:, 0:2]=Mu.squeeze(1).squeeze(-1)
        v[:, 2:6]=((Mu @ torch.transpose(Mu,-1,-2)) + Sigma).view(-1,4)
        return v # [batch,6]


    @classmethod
    def forward(cls, ctx, theta, psi):
        # We assume a Gaussian
        # We have:
        # Mu:[batch,1,2,1] and Sigma:[batch,1,2,2]
        #theta=[(Sigma)^-1 @ Mu, -0.5*(Sigma)^-1]
        #theta: batch x 6 
        #phi(t)=[t,tt^t]
        #p(t)= Gaussian(t; Mu, Sigma)

        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi

        Sigma=(-2*theta[:,2:6].view(-1,2,2))
        Sigma=(1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))).unsqueeze(1) # torch.Size([batch, 1, 2, 2])
        Mu=(Sigma @ (theta[:,0:2].view(-1,2,1)).unsqueeze(1)) # torch.Size([batch, 1, 2, 1])
        
        r=cls._expectation_psi(ctx, Mu, Sigma)
        ctx.save_for_backward(Mu, Sigma, r)
        return r # [batch, N]

    @classmethod
    def backward(cls, ctx, grad_output):
        Mu, Sigma, r = ctx.saved_tensors
        J = cls._expectation_phi_psi(ctx, Mu, Sigma) # batch,6,N
        e_phi = cls._expectation_phi(ctx, Mu, Sigma) # batch,6
        e_psi = cls._expectation_psi(ctx, Mu, Sigma) # batch,N
        J -= torch.bmm(e_phi.unsqueeze(2), e_psi.unsqueeze(1))
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None

class ContinuousSoftmax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSoftmax, self).__init__()
        self.psi = psi

    def forward(self, theta):
        return ContinuousSoftmaxFunction.apply(theta, self.psi)


class ContinuousSparsemaxFunction(torch.autograd.Function):

    @classmethod
    def _expectation_phi_psi(cls, ctx, Mu, Sigma):
        """Compute expectation of phi(t) * psi(t).T under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros((Mu.shape[0], 6, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            integral_t_times_psi=(basis_functions.integrate_t_times_psi(Mu,Sigma).squeeze(-1)).to(ctx.device)
            integral_t2_times_psi=basis_functions.integrate_t2_times_psi(Mu,Sigma).to(ctx.device)

            V[:, 0, start:offsets[j]]=integral_t_times_psi[:,:,0]
            V[:, 1, start:offsets[j]]=integral_t_times_psi[:,:,1]
            V[:, 2, start:offsets[j]]=integral_t2_times_psi[:,:,0,0]
            V[:, 3, start:offsets[j]]=integral_t2_times_psi[:,:,0,1]
            V[:, 4, start:offsets[j]]=integral_t2_times_psi[:,:,1,0]
            V[:, 5, start:offsets[j]]=integral_t2_times_psi[:,:,1,1]
            start = offsets[j]
        return V # [batch,6,N]


    @classmethod
    def _expectation_psi(cls, ctx, Mu, Sigma):
        """Compute expectation of psi under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        r = torch.zeros(Mu.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            r[:, start:offsets[j]] = basis_functions.integrate_psi(Mu, Sigma).squeeze(-2).squeeze(-1)
            start = offsets[j]
        return r # [batch,N]


    @classmethod
    def _expectation_phi(cls, ctx, Mu, Sigma):
        
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        v = torch.zeros((Mu.shape[0], 6, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        
        for j, basis_functions in enumerate(ctx.psi):
            integral_normal=basis_functions.integrate_normal(Mu, Sigma).to(ctx.device) # [batch, N, 1, 1]
            aux=(basis_functions.aux(Mu, Sigma)).to(ctx.device)

            v[:, 0, start:offsets[j]]=Mu.squeeze(-1)[:,:,0] * integral_normal.squeeze(-1).squeeze(-1)
            v[:, 1, start:offsets[j]]=Mu.squeeze(-1)[:,:,1] * integral_normal.squeeze(-1).squeeze(-1)
            v[:, 2, start:offsets[j]]=(aux * integral_normal)[:,:,0,0]
            v[:, 3, start:offsets[j]]=(aux * integral_normal)[:,:,0,1]
            v[:, 4, start:offsets[j]]=(aux * integral_normal)[:,:,1,0]
            v[:, 5, start:offsets[j]]=(aux * integral_normal)[:,:,1,1]
            start = offsets[j]
        return v # [batch,6,N]


    @classmethod
    def forward(cls, ctx, theta, psi):
        # We assume a Gaussian
        # We have:
        # Mu:[batch,1,2,1] and Sigma:[batch,1,2,2]
        #theta=[(Sigma)^-1 @ Mu, -0.5*(Sigma)^-1]
        #theta: batch x 6 
        #phi(t)=[t,tt^t]
        #p(t)= Gaussian(t; Mu, Sigma)

        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi

        Sigma=(-2*theta[:,2:6].view(-1,2,2))
        Sigma=(1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))).unsqueeze(1) # torch.Size([batch, 1, 2, 2])
        Mu=(Sigma @ (theta[:,0:2].view(-1,2,1)).unsqueeze(1)) # torch.Size([batch, 1, 2, 1])
        
        r=cls._expectation_psi(ctx, Mu, Sigma)
        ctx.save_for_backward(Mu, Sigma, r)
        return r # [batch, N]

    @classmethod
    def backward(cls, ctx, grad_output):
        Mu, Sigma, r = ctx.saved_tensors
        J = cls._expectation_phi_psi(ctx, Mu, Sigma) - cls._expectation_phi(ctx, Mu, Sigma) # batch,6,N
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None

class ContinuousSparsemax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSparsemax, self).__init__()
        self.psi = psi

    def forward(self, theta):
        return ContinuousSparsemaxFunction.apply(theta, self.psi)

# --------------------------------------------------------------
# ---- Flatten the sequence (image in continuous attention) ----
# --------------------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.attention=__C.attention
        self.gen_func=gen_func

        if str(gen_func)=='tvmax':
            self.sparsemax = partial(sparsemax, k=512)
            self.tvmax = TV2DFunction.apply

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,__C.FLAT_OUT_SIZE)

        if (self.attention=='cont-sparsemax'):
            self.transform = ContinuousSparsemax(psi=None) # use basis functions in 'psi' to define continuous sparsemax
        else:
            self.transform = ContinuousSoftmax(psi=None) # use basis functions in 'psi' to define continuous softmax
        
        device='cuda'

        # compute F and G offline for one length = 14*14 = 196
        self.Gs = [None]
        self.psi = [None]
        max_seq_len=14*14 # 196 grid features
        attn_num_basis=100 # 100 basis functions
        nb_waves=attn_num_basis
        self.psi.append([])
        self.add_gaussian_basis_functions(self.psi[1],nb_waves,device=device)


        # stack basis functions
        padding=True
        length=max_seq_len
        if padding:
            shift=1/float(2*math.sqrt(length))
            positions_x = torch.linspace(-0.5+shift, 1.5-shift, int(2*math.sqrt(length)))
            positions_x, positions_y=torch.meshgrid(positions_x,positions_x)
            positions_x=positions_x.flatten()
            positions_y=positions_y.flatten()
        else:
            shift = 1 / float(2*math.sqrt(length))
            positions_x = torch.linspace(shift, 1-shift, int(math.sqrt(length)))
            positions_x, positions_y=torch.meshgrid(positions_x,positions_x)
            positions_x=positions_x.flatten()
            positions_y=positions_y.flatten()

        positions=torch.zeros(len(positions_x),2,1).to(device)
        for position in range(1,len(positions_x)+1):
            positions[position-1]=torch.tensor([[positions_x[position-1]],[positions_y[position-1]]])

        F = torch.zeros(nb_waves, positions.size(0)).unsqueeze(2).unsqueeze(3).to(device) # torch.Size([N, 196, 1, 1])
        # print(positions.size()) # torch.Size([196, 2, 1])
        basis_functions = self.psi[1][0]
        # print(basis_functions.evaluate(positions[0]).size()) # torch.Size([N, 1, 1])

        for i in range(0,positions.size(0)):
            F[:,i]=basis_functions.evaluate(positions[i])[:]

        penalty = .01  # Ridge penalty
        I = torch.eye(nb_waves).to(device)
        F=F.squeeze(-2).squeeze(-1) # torch.Size([N, 196])
        G = F.t().matmul((F.matmul(F.t()) + penalty * I).inverse()) # torch.Size([196, N])
        if padding:
            G = G[length:-length, :]
            G=torch.cat([G[7:21,:],G[35:49,:],G[63:77,:],G[91:105,:],G[119:133,:],G[147:161,:],G[175:189,:],G[203:217,:],G[231:245,:],G[259:273,:],G[287:301,:],G[315:329,:],G[343:357,:],G[371:385,:]])
        
        self.Gs.append(G.to(device))

    def add_gaussian_basis_functions(self, psi, nb_basis, device):
        
        steps=int(math.sqrt(nb_basis))

        mu_x=torch.linspace(0,1,steps)
        mu_y=torch.linspace(0,1,steps)
        mux,muy=torch.meshgrid(mu_x,mu_y)
        mux=mux.flatten()
        muy=muy.flatten()

        mus=[]
        for mu in range(1,nb_basis+1):
            mus.append([[mux[mu-1]],[muy[mu-1]]])
        mus=torch.tensor(mus).to(device)

        sigmas=[]
        for sigma in range(1,nb_basis+1):
            sigmas.append([[0.001,0.],[0.,0.001]]) # it is possible to change this matrix
        sigmas=torch.tensor(sigmas).to(device) # in continuous softmax we have sigmas=torch.DoubleTensor(sigmas).to(device)

        assert mus.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu=mus, sigma=sigmas))

    def value_function(self, values, mask=None):
        # Approximate B * F = values via multivariate regression.
        # Use a ridge penalty. The solution is B = values * G
        # x:(batch,L,D)
        G = self.Gs[1]
        B = torch.transpose(values,-1,-2) @ G
        return B

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2),-1e9)

        if str(self.gen_func)=='tvmax':
            att = att.squeeze(-1).view(-1,14,14)
            for i in range(att.size(0)):
                att[i] = self.tvmax(att[i])
            att = self.sparsemax(att.view(-1,14*14)).unsqueeze(-1)
        
        else:
            att = self.gen_func(att.squeeze(-1), dim=-1).unsqueeze(-1)

        # compute distribution parameters
        max_seq_len=196
        length=max_seq_len

        positions_x = torch.linspace(0., 1., int(math.sqrt(length)))
        positions_x, positions_y=torch.meshgrid(positions_x,positions_x)
        positions_x=positions_x.flatten()
        positions_y=positions_y.flatten()
        positions=torch.zeros(len(positions_x),2,1).to(x.device)
        for position in range(1,len(positions_x)+1):
            positions[position-1]=torch.tensor([[positions_x[position-1]],[positions_y[position-1]]])

        # positions: (196, 2, 1)
        # positions.unsqueeze(0): (1, 196, 2, 1)
        # att.unsqueeze(-1): (batch, 196, 1, 1)
        Mu= torch.sum(positions.unsqueeze(0) @ att.unsqueeze(-1), 1) # (batch, 2, 1)
        Sigma=torch.sum(((positions @ torch.transpose(positions,-1,-2)).unsqueeze(0) * att.unsqueeze(-1)),1) - (Mu @ torch.transpose(Mu,-1,-2)) # (batch, 2, 2)
        Sigma=Sigma + (torch.tensor([[1.,0.],[0.,1.]])*1e-6).to(x.device) # to avoid problems with small values


        if (self.attention=='cont-sparsemax'):
            Sigma=9.*math.pi*torch.sqrt(Sigma.det().unsqueeze(-1).unsqueeze(-1))*Sigma

        # get `mu` and `sigma` as the canonical parameters `theta`
        theta1 = ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) @ Mu).flatten(1)
        theta2 = (-1. / 2. * (1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2)))).flatten(1)
        theta = torch.zeros(x.size(0), 6, device=x.device ) #torch.Size([batch, 6])
        theta[:,0:2]=theta1
        theta[:,2:6]=theta2

        # map to a probability density over basis functions
        self.transform.psi = self.psi[1]
        r = self.transform(theta)  # batch x nb_basis

        # compute B using a multivariate regression
        # batch x D x N
        B = self.value_function(x, mask=None)

        # (bs, nb_basis) -> (bs, 1, nb_basis)
        r = r.unsqueeze(1)  # batch x 1 x nb_basis

        # (bs, hdim, nb_basis) * (bs, nb_basis, 1) -> (bs, hdim, 1)
        # get the context vector
        # batch x values_size x 1
        context = torch.matmul(B, r.transpose(-1, -2))
        context = context.transpose(-1, -2)  # batch x 1 x values_size

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1) # don't need this for continuous attention
        
        x_atted=context.squeeze(1) # for continuous softmax/sparsemax

        x_atted = self.linear_merge(x_atted) # linear_merge is used to compute Wx
        return x_atted





# ----------------------------------------------------------------
# ---- Flatten the sequence (question and discrete attention) ----
# ----------------------------------------------------------------
# this is also used to flatten the image features with discrete attention
class AttFlatText(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(AttFlatText, self).__init__()
        self.__C = __C

        self.gen_func=gen_func

        if str(gen_func)=='tvmax':
            self.sparsemax = partial(sparsemax, k=512)
            self.tvmax = TV2DFunction.apply

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,__C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2),-1e9)

        if str(self.gen_func)=='tvmax':
            att = att.squeeze(-1).view(-1,14,14)
            for i in range(att.size(0)):
                att[i] = self.tvmax(att[i])
            att = self.sparsemax(att.view(-1,14*14)).unsqueeze(-1)
        
        else:
            att = self.gen_func(att.squeeze(-1), dim=-1).unsqueeze(-1)
        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, gen_func=torch.softmax):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=__C.WORD_EMBED_SIZE)

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.attention=__C.attention #added this 


        #if __C.USE_IMG_POS_EMBEDDINGS:
        #    self.img_pos_x_embeddings = nn.Embedding(num_embeddings=14, embedding_dim=int(__C.HIDDEN_SIZE/2))
        #    torch.nn.init.xavier_uniform_(self.img_pos_x_embeddings.weight)
        #    self.img_pos_y_embeddings = nn.Embedding(num_embeddings=14, embedding_dim=int(__C.HIDDEN_SIZE/2))
        #    torch.nn.init.xavier_uniform_(self.img_pos_y_embeddings.weight)
        #    self.use_img_pos_embeddings = __C.USE_IMG_POS_EMBEDDINGS

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True)

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE)

        self.gen_func=gen_func
        self.backbone = MCA_ED(__C, gen_func)

        if (self.attention=='discrete'):
            self.attflat_img = AttFlatText(__C, self.gen_func)
        else: # use continuous attention 
            self.attflat_img = AttFlat(__C, self.gen_func)

        self.attflat_lang = AttFlatText(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        #if self.use_img_pos_embeddings:
        #    for i in range(img_feat.size(0)):
        #        pos = torch.LongTensor(np.mgrid[0:14,0:14]).cuda()
        #        img_feat[i]+=torch.cat([self.img_pos_x_embeddings(pos[0].view(-1)), self.img_pos_y_embeddings(pos[1].view(-1))],1)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask)

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask)

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask)

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature),dim=-1) == 0).unsqueeze(1).unsqueeze(2)
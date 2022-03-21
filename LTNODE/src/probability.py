import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from scipy.special import gamma
from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform

from src.utils import torch_onehot

#Line 202,105
def gumbel_softmax(log_prob_map, temperature, eps=1e-20):
    """Note that inputs are logprobs"""
    u = log_prob_map.new(log_prob_map.shape).uniform_(0, 1)
    g = -torch.log(-torch.log(u + eps) + eps)
    softmax_in = (log_prob_map + eps) + g
    y = F.softmax(softmax_in / temperature, dim=1)
    y_hard = torch.max(y, dim=1)[1]
    y_hard = torch_onehot(y_hard, y.shape[1]).type(y.type())
    return (y_hard - y).detach() + y


def gumbel_sigmoid(prob_map, temperature, eps=1e-20):
    U = prob_map.new(prob_map.shape).uniform_(0, 1)
    sigmoid_in = torch.log(prob_map + eps) - torch.log(1 - prob_map + eps) + torch.log(U + eps) - torch.log(1 - U + eps)
    y = torch.sigmoid(sigmoid_in / temperature)
    y_hard = torch.round(y)
    return (y_hard - y).detach() + y


class diag_w_Gauss_loglike(object):
    def __init__(self, μ=0, σ2=1, multiplier=1):
        super(diag_w_Gauss_loglike, self).__init__()
        self.μ = μ
        self.σ2 = σ2
        self.dist = Normal(self.μ, self.σ2)
        self.multiplier = multiplier

    def __call__(self, w):
        log_pW = self.dist.log_prob(w)
        return log_pW.sum() * self.multiplier

    def summary(self):
        return {"name": "diag_w_Gauss", "μ": self.μ, "σ2": self.σ2, "mul": self.multiplier}


class homo_Gauss_mloglike(nn.Module):
    def __init__(self, Ndims=1, sig=None):
        super(homo_Gauss_mloglike, self).__init__()
        if sig is None:
            self.log_std = nn.Parameter(torch.zeros(Ndims))
        else:
            self.log_std = nn.Parameter(torch.ones(Ndims) * np.log(sig), requires_grad=False)

    def forward(self, mu, y, model_std=None):
        sig = self.log_std.exp().clamp(min=1e-4)
        #if math.isnan(sig)
        #print("homo_Gauss_mloglike",sig)
        if model_std is not None:
            #print("homo_Gauss_mloglik jdvhjkdjdjfkjfkle",sig.shape,model_std.shape)
            sig = (sig**2 + model_std**2)**0.5
        #print("homo_Gauss_mloglike",sig.shape)
        dist = Normal(mu, sig)
        #print("homo_Gauss_mloglike  neg_pred_loglike",sig)
        return -dist.log_prob(y)


class pMOM_loglike(object):
    def __init__(self, r=1, τ=0.348, σ2=1.0, multiplier=1):
        super(pMOM_loglike).__init__()
        self.r = r
        self.τ = τ
        self.σ2 = σ2
        self.multiplier = multiplier

        self.d_p = -np.log(1 if self.r == 1 else 3 if self.r == 2 else 15)

    def __call__(self, W):
        p = W.numel()

        log_pW = self.d_p
        log_pW -= self.r * p * np.log(self.τ * self.σ2)
        log_pW += torch.sum(self.r * torch.log(W ** 2))
        log_pW += torch.distributions.normal.Normal(0, np.sqrt(self.τ * self.σ2)).log_prob(W).sum()

        return log_pW * self.multiplier

    def summary(self):
        return {"name": "pMOM", "r": self.r, "τ": self.τ, "σ2": self.σ2, "mul": self.multiplier}


class piMOM_loglike(object):
    def __init__(self, r=1, τ=0.348, σ2=1.0, multiplier=1):
        super(piMOM_loglike).__init__()
        self.r = r
        self.τ = τ
        self.σ2 = σ2
        self.multiplier = multiplier

        self.log_C = (-r + 1/2) * np.log(τ * σ2) + np.log(gamma(r - 1/2))

    def __call__(self, W):
        p = W.numel()

        log_pW = -p * self.log_C
        log_pW += torch.sum(-self.τ * self.σ2 / (W ** 2))
        log_pW += torch.sum(-self.r * torch.log(W ** 2))

        return log_pW * self.multiplier

    def summary(self):
        return {"name": "piMOM", "r": self.r, "τ": self.τ, "σ2": self.σ2, "mul": self.multiplier}


class peMOM_loglike(object):
    def __init__(self, r=None, τ=0.348, σ2=1.0, multiplier=1):
        super(peMOM_loglike).__init__()
        self.τ = τ
        self.σ2 = σ2
        self.multiplier = multiplier

        self.log_C = 0.5 * np.log(2 * np.pi * σ2 * τ) - (2/σ2)**0.5

    def __call__(self, W):
        p = W.numel()

        log_pW = -p * self.log_C
        log_pW += torch.sum(-W**2/(2 * self.σ2 * self.τ))
        log_pW += torch.sum(-self.τ/(W ** 2))

        return log_pW * self.multiplier

    def summary(self):
        return {"name": "pMOM", "τ": self.τ, "σ2": self.σ2, "mul": self.multiplier}

import torch
import torch.nn as nn
import torch.nn.functional as F


class depth_gamma(nn.Module):
    def __init__(self, prior_probs, prior_logprobs=None, cuda=True):
        # TODO: add option of specifying prior in terms of log_probs
        super(depth_gamma, self).__init__()

        self.prior_probs = torch.Tensor(prior_probs)
        #print("self.prior_probs",self.prior_probs)
        #assert self.prior_probs.sum().item() - 1 < 1e-6
        self.dims = self.prior_probs.shape[0]
        if prior_logprobs is None:
            self.logprior = self.prior_probs.log()
        else:
            self.logprior = torch.Tensor(prior_logprobs)
            self.prior_probs = self.logprior.exp()
            assert self.prior_probs.sum().item() - 1 < 1e-6

        #print("self.logprior",self.logprior)
        self.current_posterior = None
        self.current_posterior_pdf = None

        self.cuda = cuda
        if self.cuda:
            self.to_cuda()

    def to_cuda(self):
        self.prior_probs = self.prior_probs.cuda()
        self.logprior = self.logprior.cuda()

    @staticmethod
    def get_w_joint_loglike(prior_loglikes, act_vec, y, f_neg_loglike, N_train):
        """Note that if we average this to get exact joint, then all batches need to be the same size.
        Alternatively you can weigh each component with its batch size."""
        batch_size = act_vec.shape[1]
        depth = act_vec.shape[0]

        #print("act_vec.shape",act_vec.shape)
        repeat_dims = [depth] + [1 for i in range(1, len(y.shape))]
        y_expand = y.repeat(*repeat_dims)  # targets are same across acts -> interleave
        #let y = [2,3,4] after y.repeat(3), [2,3,4,2,3,4,2,3,4]
        act_vec_flat = act_vec.view(depth*batch_size, -1)  # flattening results in batch_n changing first
        #from 6,256,10 to 6x256,10.. I guess, after placing 256x10 of first dimension, second 256x10 are placed
        #print(act_vec.shape,act_vec_flat.shape)
        #print("probability act_Vect_Flat y_expand", act_vec_flat.shape,y_expand.shape)
        loglike_per_act = -f_neg_loglike(act_vec_flat, y_expand).view(depth, batch_size)
        #print("loglike_per_act",loglike_per_act)
        #print("probability loglike_per_act ", loglike_per_act,"N_train",N_train)
        #f_neg_loglike is cross entropy which takes logits as input.. 

        joint_loglike_per_depth = (N_train / batch_size) * loglike_per_act.sum(dim=1) #+ prior_loglikes  # (depth)
        #joint_loglike_per_depth = loglike_per_act.sum(dim=1) #+ prior_loglikes  # (depth)
        return joint_loglike_per_depth

    def get_marg_loglike(self, joint_loglike_per_depth):
        log_joint_with_depth = joint_loglike_per_depth #+ self.logprior
        log_marginal_over_depth = torch.logsumexp(log_joint_with_depth, dim=0)
        return log_marginal_over_depth

    def get_depth_log_posterior(self, joint_loglike_per_depth, log_marginal_over_depth=None):
        if log_marginal_over_depth is None:
            log_marginal_over_depth = self.get_marg_loglike(joint_loglike_per_depth)
        log_joint_with_depth = joint_loglike_per_depth# + self.logprior
        log_depth_posteriors = log_joint_with_depth - log_marginal_over_depth
        return log_depth_posteriors

    @staticmethod
    def marginalise_d_predict(act_vec, d_posterior, depth=None, softmax=False, get_std=False):
        """ Predict while marginalising d with given distribution."""
        # TODO: switch to logprobs and log q
        assert not (softmax and get_std)
        if softmax:
            preds = F.softmax(act_vec, dim=2)
            #print(preds)
            #print("in probability",preds[:,0,:])
        else:
            preds = act_vec

        q = d_posterior.clone().detach()
        while len(q.shape) < len(act_vec.shape):
            q = q.unsqueeze(1)
        #print("marginalize the predit")
        mc = q.shape[0]
        #print("marginalize the predict",q.shape,preds.shape,(q * (preds)).shape)
        if get_std:
            pred_mu = ((preds)/mc).sum(dim=0)
            model_var = (((preds**2)).sum(dim=0)/mc - pred_mu**2)
            #model_var = ((((preds)) - pred_mu).sum(0))**2/mc
            print("marginalise_d_predict  mean and variance",pred_mu[0:3],model_var[0:3])
            return pred_mu, model_var.pow(0.5)

        weighed_preds =  preds/mc
        '''

        if get_std:
            pred_mu = (preds/mc).sum(dim=0)
            model_var = ((preds**2)/mc).sum(dim=0) - pred_mu**2
            return pred_mu, model_var.pow(0.5)
        
        weighed_preds = preds/mc
        '''
        return weighed_preds.sum(dim=0)

class depth_uniform():
    def __init__(self,low,high):
        super(depth_uniform,self).__init__()
        self.low= low
        self.high = high

    def get_samples(self,n_samples):
        m = Uniform(self.low,self.high)
        self.samples = m.sample([n_samples]).sort(0)[0] #sort returns values with indeces, we only need values
        return self.samples

    def marginalise_d_predict(self,act_vec, n_samples, depth=None, softmax=False, get_std=False):
        """ Predict while marginalising d with given distribution."""
        # TODO: switch to logprobs and log q
        assert not (softmax and get_std)
        if softmax:
            preds = F.softmax(act_vec, dim=2)

        else:
            preds = act_vec
        mc = n_samples
        if get_std:
            pred_mu = (preds/mc).sum(dim=0)
            model_var = ((preds**2)/mc).sum(dim=0) - pred_mu**2

            return pred_mu, model_var.pow(0.5)
        
        weighed_preds = preds/mc
        return weighed_preds.sum(dim=0)

class depth_gamma_VI(depth_gamma):

    def __init__(self, prior_probs, cuda=True, eps=1e-35):
        super(depth_gamma_VI, self).__init__(prior_probs, None, cuda)

        self.q_a = nn.Parameter(torch.zeros(1)+1, requires_grad=True)
        self.q_b = nn.Parameter(torch.zeros(1)+1, requires_grad=True)
        self.prior = torch.Tensor(prior_probs)
        self.lik = None
        self.KL = None
        self.eps = eps
        self.samples = None
        self.Train = True
        if cuda:
            self.to_cuda_VI()

    def to_cuda_VI(self):

        self.q_a.data = self.q_a.data.cuda()
        self.q_b.data = self.q_b.data.cuda()

    def get_samples(self,n_samples):
        #print(self.Train)
        if self.Train == False:
            m = Gamma(self.q_a,self.q_b)
        else:
            m = Uniform(torch.Tensor([0.000001]),torch.Tensor([3.]))#Gamma(self.q_a,self.q_b)#3 for paper images,and syntheitc.
        self.samples = m.rsample([n_samples]).sort(0)[0] #sort returns values with indeces, we only need values
        return self.samples

    def get_pdf(self,samples):

        m = Gamma(self.q_a,self.q_b)
        return (m.log_prob(samples)).exp()

    '''
    def get_q_logprobs(self):
        """Get logprobs of each depth configuration"""
        return F.log_softmax(self.q_logits, dim=0)
    '''
    def get_q_probs(self):
        """Get probs of each depth configuration"""
        #return F.softmax(self.get_pdf(self.samples), dim=0)
        return self.get_pdf(self.samples)
   
    def get_KL(self):
        """KL between categorical distributions"""
        #log_q = self.get_q_logprobs()
        #q = self.get_q_probs().clamp(min=self.eps, max=(1 - self.eps))
        #log_p = self.logprior
        #print(log_p)
        t1 = self.q_a*torch.log(self.q_b) - self.prior[0]*torch.log(self.prior[1])
        t2 = torch.lgamma(self.prior[0]) - torch.lgamma(self.q_a)
        t3 = torch.digamma(self.q_a)*(self.q_a - self.prior[0]) - (self.q_a - self.prior[0])*torch.log(self.q_b)
        t4 = (torch.exp(torch.lgamma(self.q_a+1))/torch.exp(torch.lgamma(self.q_a))) *(self.prior[1]/self.q_b) -self.q_a
        KL = t1 + t2 + t3 + t4
        return KL

    def get_E_loglike(self, joint_loglike_per_depth):
        """Calculate ELBO with deterministic expectation."""
        
        q = self.get_q_probs()
        
        #print("probability ",joint_loglike_per_depth)
        #print("q[:,0], joint_loglike_per_depth",q[:,0].shape,joint_loglike_per_depth.shape)
        
        E_loglike = (q[:,0]* joint_loglike_per_depth).sum(dim=0) #multiply with q[:,0], if after applying softmax
        #E_loglike = (joint_loglike_per_depth).sum(dim=0)
        
        return E_loglike

    def estimate_ELBO(self, prior_loglikes, act_vec, y, f_neg_loglike, N_train, Beta=1):
        """Estimate ELBO on logjoint of data and network weights"""
        #print("estimate_ELBO.....................")
        joint_loglike_per_depth = depth_gamma.get_w_joint_loglike(prior_loglikes, act_vec, y,
                                                                        f_neg_loglike, N_train)
        # print('Elbo joint loglike per depth', joint_loglike_per_depth)
        #print("act_vec.shape[0] act_vec.shape[1] ",act_vec.shape[0],act_vec.shape[1])
        E_loglike = self.get_E_loglike(joint_loglike_per_depth)/(act_vec.shape[0])#act_vec.shape[1]#N_train#/act_vec.shape[0]  # act_vec.shape[0] gives number of posterior samples
        KL = self.get_KL()
        
        self.lik = E_loglike.item()
        self.KL = KL.item()
        #print("E_loglike,KL ",E_loglike,KL)
        return E_loglike - (0.05*KL*N_train)/(1.)#/act_vec.shape[1] #0.1 for image
    def estimate_lik_kl(self, prior_loglikes, act_vec, y, f_neg_loglike, N_train, Beta=1):
        return self.lik,self.KL


    def q_predict(self, act_vec, depth=None, softmax=False):
        """Predict marginalising depth with approximate posterior. Currently this will only support classification"""
        return depth_categorical.marginalise_d_predict(act_vec, self.get_q_probs(), depth=depth, softmax=softmax)

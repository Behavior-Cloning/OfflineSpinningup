import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys
import os
base_dir=os.path.abspath(__file__)
dir=os.path.abspath(__file__)
for ii in range(3):
    dir=os.path.dirname(dir)
sys.path.append(dir)

from offlinespinup.utils.mlp_infrastructure import build_mlp

LOG_STD_MAX=2
LOG_STD_MIN=-20

class SquashedGaussianActor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_sizes=[256,256],act_limit=1.0,activation=nn.ReLU,output_activation=nn.Identity):
        super().__init__()
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.net=build_mlp([obs_dim]+list(hidden_sizes)+[2*act_dim],activation,output_activation)
        self.act_limit=act_limit

    def forward(self,obs,deterministic=False,with_logp_pi=True,num_samples=1):
        B,O=tuple(obs.shape)
        A=self.act_dim
        net_out=self.net(obs)
        mu,log_std=net_out.chunk(2,dim=1)# (B,A) (B,A)
        log_std=torch.clamp(log_std,LOG_STD_MIN,LOG_STD_MAX)
        std=torch.exp(log_std)

        dist=Normal(mu,std)
        if deterministic:
            pi_action=mu
        else:
            pi_action=dist.rsample((num_samples,))
            assert tuple(pi_action.shape)==(num_samples,B,A)
        
        if with_logp_pi:
            logp_pi = dist.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
            if deterministic:
                assert tuple(logp_pi.shape)==(B,)
            else:
                assert tuple(logp_pi.shape)==(num_samples,B)
        else:
            logp_pi=None
        
        pi_action=torch.tanh(pi_action)
        pi_action=self.act_limit*pi_action
        return pi_action,logp_pi# determinstic:(B,A) (B,) else:(N,B,A) (N,B)
    
    def get_action(self,obs,deterministic):
        if not isinstance(obs,torch.Tensor):
            device=self.net[0].weight.device
            obs=torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
        with torch.no_grad():
            a,_=self.forward(obs,deterministic,with_logp_pi=False,num_samples=1)
            if deterministic:
                return a[0].detach().cpu().numpy()
            else:
                return a[0,0].detach().cpu().numpy()

    def atanh(self,x):
        return 0.5*torch.log((1.0+x)/(1.0-x+1e-6))

    def compute_action_logp(self,obs,act):
        act=act.clone()
        if act.dim()==2:
            act.unsqueeze_(0)
        act_normal=self.atanh(act)
        net_out=self.net(obs)
        mu,log_std=net_out.chunk(2,dim=1)# (B,A) (B,A)
        log_std=torch.clamp(log_std,LOG_STD_MIN,LOG_STD_MAX)
        std=torch.exp(log_std)

        dist=Normal(mu,std)

        logp_pi = dist.log_prob(act_normal).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - act_normal - F.softplus(-2*act_normal))).sum(axis=-1)
        assert logp_pi.dim()==2
        return logp_pi


            
if __name__=='__main__':
    O,A,B=5,2,8
    actor=SquashedGaussianActor(O,A)
    obs=torch.rand((B,O))
    a,logp_pi=actor(obs,deterministic=True)
    # print(a.shape,logp_pi.shape)
    # print(actor.get_action([1.0]*O))
    # print('\nnot deterministic')
    # for i in range(5):
    #     print(actor.get_action([1.0]*O,deterministic=False))

    # a,logp_pi=actor(obs,deterministic=False,num_samples=10)
    # print(a.shape,logp_pi.shape)
    # print(a[:,1,:])

    print(logp_pi)
    print(a)
    logp_pi=actor.compute_action_logp(obs,a)
    print(logp_pi)
    print(a)
from copy import deepcopy
import numpy as np
import torch
import itertools
from torch import nn
from torch.nn import functional as F
from offlinespinup.utils.actor import SquashedGaussianActor
from offlinespinup.utils.critic import Critic

class SACpolicy:
    
    def __init__(self,obs_dim,act_dim,
    a_kwargs=dict(hidden_sizes=[256,256]),
    c_kwargs=dict(ensemble_size=2,hidden_sizes=[256,256]),
    alpha=0.2,tau=0.005,num_samples=1,gamma=0.99,auto_temp=False):
        self.tau=tau
        self.num_samples=num_samples
        self.gamma=gamma
        self.actor=SquashedGaussianActor(obs_dim,act_dim,**a_kwargs)
        self.critic=Critic(obs_dim,act_dim,**c_kwargs)
        self.actor_targ=deepcopy(self.actor)
        self.critic_targ=deepcopy(self.critic)
        self.auto_temp=auto_temp
        self.target_entropy=-act_dim
        self.log_alpha=nn.Parameter(torch.tensor(np.log(alpha)))
        if auto_temp:
            self.log_alpha.requires_grad_(True)
        else:
            self.log_alpha.requires_grad_(False)
        
        for p in itertools.chain(self.actor_targ.parameters(),self.critic_targ.parameters()):
            p.requires_grad=False

    def compute_pi_loss(self,data):
        alpha=torch.exp(self.log_alpha.detach())
        o=data['obs']
        pi_action,logp_pi=self.actor(o,num_samples=self.num_samples)#(N,B,A) (N,B)
        logpi=torch.mean(logp_pi,dim=0)#(B,)

        qs=self.critic(o,pi_action)#(N,E,B)
        q,_=torch.min(qs,dim=1)#(N,B)
        v=torch.mean(q,dim=0)#(B,)

        pi_loss=-(v-alpha*logpi).mean()

        pi_info=dict(LogPi=logpi.detach().cpu().numpy())

        return pi_loss,pi_info

    def compute_q_loss(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        alpha=torch.exp(self.log_alpha.detach())

        qs=self.critic(o,a).squeeze(0)#(1,E,B)->(E,B)
        E,B=tuple(qs.shape)
        A=a.shape[1]
        N=self.num_samples
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(o2,num_samples=self.num_samples)#(N,B,A) (N,B)
            assert tuple(a2.shape)==(N,B,A)
            assert tuple(logp_a2.shape)==(N,B)
            logpi=logp_a2.mean(dim=0)#(B,)

            # Target Q-values
            qs_pi_targ=self.critic_targ(o2,a2)#(N,E,B)
            assert tuple(qs_pi_targ.shape)==(N,E,B)
            v_pi_targ = torch.min(qs_pi_targ,dim=1)[0].mean(0)#(B,)
            backup = r + self.gamma * (1 - d) * (v_pi_targ - alpha * logpi)

        # MSE loss against Bellman backup
        loss_q=((qs-backup.unsqueeze(0))**2).mean(dim=-1).sum()

        # Useful info for logging
        q_info = dict(QVals=qs.detach().cpu().numpy())

        return loss_q, q_info

    def compute_alpha_loss(self,data):
        """this function is still under test"""
        if self.auto_temp:
            with torch.no_grad():
                o=data['obs']
                _,logpi=self.actor(o,self.num_samples)
                logpi=logpi.mean(0)+self.target_entropy
            loss_alpha=-(self.log_alpha*logpi).mean()
            alpha_info=dict(Alpha=torch.exp(self.log_alpha).detach().cpu().item(),
                            LogAlpha=self.log_alpha.detach().cpu().item())
            return loss_alpha,alpha_info

    @torch.no_grad()
    def syn_weight(self):
        for p,pt in zip(self.actor.parameters(),self.actor_targ.parameters()):
            pt.copy_(pt.data*(1.0-self.tau)+self.tau*p.data)
        for p,pt in zip(self.critic.parameters(),self.critic_targ.parameters()):
            pt.copy_(pt.data*(1.0-self.tau)+self.tau*p.data)
  
    def to_device(self,device='cuda:0'):
        self.actor.to(device),self.critic.to(device),self.log_alpha.to(device)
        self.actor_targ.to(device),self.critic_targ.to(device)
    
    def get_action(self,obs,deterministic):
        return self.actor.get_action(obs,deterministic)

if __name__=='__main__':
    O,A,B=20,6,256

    data={}
    data['obs'], data['act'], data['rew'], data['obs2'], data['done']=torch.randn((B,O)),\
        torch.randn((B,A)),torch.randn((B,)),torch.randn((B,O)),torch.rand((B))
    
    sac=SACpolicy(O,A)
    pi_loss,pi_info=sac.compute_pi_loss(data)
    q_loss,q_info=sac.compute_q_loss(data)
    print(pi_loss)
    print(q_loss)
    

import torch
import torch.nn as nn
from offlinespinup.utils.mlp_infrastructure import Ensemble_Linear,build_mlp

class Critic(nn.Module):
    def __init__(self,obs_dim,act_dim,ensemble_size,hidden_sizes,activation=nn.ReLU,output_activation=nn.Identity):
        super().__init__()
        self.ensemble_size=ensemble_size
        sizes=[obs_dim+act_dim]+list(hidden_sizes)+[1]
        layers=[]
        for i in range(len(sizes)-1):
            act=activation if i <len(sizes)-2 else output_activation
            layers+=[Ensemble_Linear(sizes[i],sizes[i+1],ensemble_size),act()]
        self.net=nn.Sequential(*layers)
    def forward(self,obs,act):
        # obs: (B,O) act: (N,B,A)
        o,a=obs.clone(),act.clone()
        if a.dim()==2:
            a.unsqueeze_(0)

        N,B,A=tuple(a.shape)
        B,O=tuple(o.shape)
        E=self.ensemble_size

        o=o.unsqueeze(0).repeat_interleave(N,dim=0)
        xs=torch.cat([o,a],dim=-1)# (N,B,A+O)
        assert tuple(xs.shape)==(N,B,A+O)

        qs=[]
        for x in xs: # x: (B,A)
            x1=x.unsqueeze(0).repeat_interleave(self.ensemble_size,dim=0)
            assert tuple(x1.shape)==(E,B,A+O)
            qs.append(self.net(x1).squeeze(-1).unsqueeze(0))
        qs=torch.cat(qs,dim=0)
        assert tuple(qs.shape)==(N,E,B)
        return qs
    


if __name__=='__main__':
    N,E,B,O,A=10,7,8,4,2
    c=Critic(O,A,E,[256,256])
    print(c.net)
    obs=torch.randn(B,O)
    act=torch.randn(B,A)
    q=c(obs,act)
    print(q)
    print(q.shape)

    for i in range(E):
        net=build_mlp([O+A]+[256,256]+[1])
        print(net(torch.cat([obs,act],dim=1)).squeeze(-1))
    print(net)
    
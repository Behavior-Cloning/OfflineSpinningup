import torch
import torch.nn as nn
import numpy as np

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

@torch.no_grad()
def torch_default_init(m):
    if m.weight.dim()==3:
        I,O=m.weight.shape[1:]
        wei,b=[],[]
        for i in range(m.weight.shape[0]):
            fc=nn.Linear(I,O)
            # print(fc.weight.data)
            wei.append(fc.weight.data.clone().t().unsqueeze(0))
            # print(fc.bias.data)
            b.append(fc.bias.data.clone().unsqueeze(0).unsqueeze(0))
        b=torch.cat(b,dim=0)
        wei=torch.cat(wei,dim=0)
        # print(wei.shape)
        # print(b.shape)
        m.weight.data.copy_(wei)
        m.bias.data.copy_(b)

@torch.no_grad()
def truncated_normal_init(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    input_dim = m.in_features
    truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
    nn.init.zeros_(m.bias)

def build_mlp(sizes,activation=nn.ReLU,output_activation=nn.Identity):
    net=[]
    for i in range(len(sizes)-1):
        act=activation if i<len(sizes)-2 else output_activation
        net+=[nn.Linear(sizes[i],sizes[i+1]),act()]
    return nn.Sequential(*net)

class Ensemble_Linear(nn.Module):
    __constants__ = ['in_features', 'out_features','ensemble_size']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self,in_features:int,out_features:int,ensemble_size:int,weight_decay:float=0.0,
    init_method=torch_default_init):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.ensemble_size=ensemble_size
        self.weight=nn.Parameter(torch.Tensor(ensemble_size,in_features,out_features))
        self.weight_decay=weight_decay
        self.bias=nn.Parameter(torch.Tensor(ensemble_size,1,out_features))
        self.apply(init_method)

    def forward(self,x):
        assert x.shape[0]==self.ensemble_size and x.dim()==3
        return torch.bmm(x,self.weight)+self.bias
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ensemble_size={self.ensemble_size}'
    


if __name__=="__main__":
    I,O=256,1
    E=1
    efc=Ensemble_Linear(I,O,E)
    print(efc.weight)
    fc=nn.Linear(I,O)
    print(fc.weight)

    print('bias:\n')
    print(efc.bias)
    print(fc.bias)
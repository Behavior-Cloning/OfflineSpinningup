import numpy as np

def statistic_records(x:list):
    x=np.array(x)
    return x.mean(),x.std(),x.max(),x.min()

class MetaDataHolder:
    def __init__(self,):
        self.N=self.Std=self.Min=self.Max=self.Mean=0.0
        self.Min=np.inf
        self.Max=-np.inf
    def incremental_estimate(self,x):
        """in model rollout, the info will return array with same dims but different shape, seems np.xxx can't deal with this issue"""
        if isinstance(x,np.ndarray):
            h_mu=self.Mean
            h_n=self.N
            a_mu=np.mean(x)
            a_n=np.prod(x.shape)

            self.Mean=(h_mu*h_n+a_mu*a_n)/(a_n+h_n)
            self.Min=np.min([self.Min,np.min(x)])
            self.Max=np.max([self.Max,np.max(x)])
            self.Std=np.sqrt(
                (h_n*(self.Std**2+(self.Mean-h_mu)**2)+a_n*(np.var(x)+(self.Mean-a_mu)**2))
                                    /(h_n+a_n)
            )
            self.N+=a_n
        elif isinstance(x,(int,float)):
            h_mu=self.Mean
            h_n=self.N

            self.Mean=(h_mu*h_n+x)/(1.0+h_n)
            self.Min=np.min([self.Min,x])
            self.Max=np.max([self.Max,x])
            self.Std=np.sqrt(
                (h_n*(self.Std**2+(self.Mean-h_mu)**2)+(self.Mean-x)**2)
                                    /(h_n+1.0)
            )
            self.N+=1.0
        else:
            raise NotImplementedError('Currently only ndarray, int, float are supported')
    
    def reset(self):
        self.__init__()


class EasyLogger:
    def __init__(self):
        self.epoch_dict={}
    
    def store(self,**kwargs):
        for k,v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k]=[]
            self.epoch_dict[k].append(v)
    
    def dump_records(self):
        for k,v in self.epoch_dict.items():
            mu,std,ub,lb=statistic_records(v)
            print(f'{k} ====> [{lb:.3f}, {ub:.3f}], mu: {mu:.3f}, std: {std:.3f}')
            self.epoch_dict[k]=[]
    
    def log(self,*msg):
        print(*msg)


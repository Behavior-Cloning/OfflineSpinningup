import numpy as np

def statistic_records(x:list):
    x=np.array(x)
    return x.mean(),x.std(),x.max(),x.min()

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
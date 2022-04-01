import torch
from offlinespinup.utils.dynamic_model import Ensemble_Dynamic_Model

class Fake_Env:
    def __init__(self,dynamic_model:Ensemble_Dynamic_Model):
        self.dynamic_model=dynamic_model
    
    def rollout(self,o0,actor,num_steps=5):
        E=self.dynamic_model.network_size
        if not isinstance(o0,torch.Tensor):
            o0_tensor=torch.tensor(o0,dtype=torch.float32,device=self.dynamic_model.device)
        else:
            o0_tensor=torch.clone(o0,dtype=torch.float32,device=self.dynamic_model.device)
        
        

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import sys
import os
base_dir=os.path.abspath(__file__)
dir=os.path.abspath(__file__)
for ii in range(3):
    dir=os.path.dirname(dir)
sys.path.append(dir)

from offlinespinup.utils.mlp_infrastructure import *


class StandardScaler(nn.Module):

    def __init__(self, input_size):
        super(StandardScaler, self).__init__()
        self.register_buffer('std', torch.ones(1, input_size))
        self.register_buffer('mu', torch.zeros(1, input_size))

    def fit(self, data):
        std, mu = torch.std_mean(data, dim=0, keepdim=True)
        std[std < 1e-12] = 1
        self.std.data.mul_(0.0).add_(std)
        self.mu.data.mul_(0.0).add_(mu)

    def transform(self, data):
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        return self.std * data + self.mu



class Ensemble_Model(nn.Module):
    def __init__(self,obs_dim,act_dim,rew_dim=1,ensemble_size=7,hidden_dim=200,
    lr=1e-3,deterministic=False):
        super().__init__()
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.rew_dim=rew_dim
        self.ensemble_size=ensemble_size
        self.hidden_dim=hidden_dim
        self.out_dim=2*(obs_dim+rew_dim) if not deterministic else obs_dim+act_dim
        self.deterministic=deterministic

        self.nn1=Ensemble_Linear(obs_dim+act_dim,hidden_dim,ensemble_size,weight_decay=0.000025,init_method=truncated_normal_init)
        self.nn2=Ensemble_Linear(hidden_dim,hidden_dim,ensemble_size,weight_decay=0.00005,init_method=truncated_normal_init)
        self.nn3=Ensemble_Linear(hidden_dim,hidden_dim,ensemble_size,weight_decay=0.000075,init_method=truncated_normal_init)
        self.nn4=Ensemble_Linear(hidden_dim, hidden_dim, ensemble_size, weight_decay=0.000075,init_method=truncated_normal_init)
        self.nn5=Ensemble_Linear(hidden_dim,self.out_dim,ensemble_size,weight_decay=0.0001,init_method=truncated_normal_init)
        self.max_logvar = nn.Parameter(torch.ones(1, obs_dim+rew_dim).float() * 0.5, requires_grad=False)
        self.min_logvar = nn.Parameter(torch.ones(1, obs_dim+rew_dim).float() * -10, requires_grad=False)
        self.swish=Swish()
        
        self.optimizer=Adam([{'params':self.nn1.parameters(),'weight_decay':self.nn1.weight_decay},
            {'params':self.nn2.parameters(),'weight_decay':self.nn2.weight_decay},
            {'params':self.nn3.parameters(),'weight_decay':self.nn3.weight_decay},
            {'params':self.nn4.parameters(),'weight_decay':self.nn4.weight_decay},
            {'params':self.nn5.parameters(),'weight_decay':self.nn5.weight_decay}],lr=lr)
        
    def forward(self,x,ret_log_var=False):
        nn1_output=self.swish(self.nn1(x))
        nn2_output=self.swish(self.nn2(nn1_output))
        nn3_output=self.swish(self.nn3(nn2_output))
        nn4_output=self.swish(self.nn4(nn3_output))
        nn5_output=self.nn5(nn4_output)

        mean,logvar=nn5_output.chunk(2,dim=-1)
        logvar=logvar-F.softplus(logvar-self.max_logvar)
        logvar=logvar+F.softplus(self.min_logvar-logvar)
        if self.deterministic:
            return mean,None

        if ret_log_var:
            return mean,logvar
        else:
            return mean,torch.exp(logvar)

    def compute_loss(self,mean,logvar,labels):
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss_inv = (torch.pow(mean - labels, 2) * inv_var).mean(dim=(1, 2))
        var_loss = logvar.mean(dim=(1, 2))
        mse_loss = torch.pow(mean - labels, 2).mean(dim=(1, 2))
        total_loss = mse_loss_inv.sum() + var_loss.sum()
        return total_loss, mse_loss


class Ensemble_Dynamic_Model(nn.Module):
    def __init__(self,obs_dim,act_dim,reward_dim=1,network_size=7,elite_size=5,hidden_dim=200,
    lr=1e-3,deterministic=False,device='cuda:0'):
        super().__init__()
        self.nets=Ensemble_Model(obs_dim,act_dim,reward_dim,network_size,hidden_dim,lr,deterministic)
        self.scaler=StandardScaler(obs_dim+act_dim)
        self.device=device

        self.elite_size=elite_size
        self.to(device)

    def train(self,obs,act,obs2,rew,batch_size=32,holdout_ratio=0.1,max_holdout_num=10000,max_epoch=10):
        inputs=np.concatenate((obs,act),axis=-1)
        targets=np.concatenate((obs2-obs,rew[:,None] if rew.ndim==1 else rew),axis=-1)
        self.scaler.fit(torch.tensor(inputs,dtype=torch.float32,device=self.device))

        total_num=obs.shape[0]
        num_holdout=int(min(total_num*holdout_ratio,max_holdout_num))
        num_train=total_num-num_holdout

        E=self.nets.ensemble_size
        O=obs.shape[-1]

        permutation = np.random.permutation(inputs.shape[0])
        train_inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        train_targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        # holdout_inputs = np.tile(holdout_inputs[None], [E, 1, 1])
        # holdout_targets = np.tile(holdout_targets[None], [E, 1, 1])

        idxs=np.argsort(np.random.rand(E,num_train),axis=-1)
        grad_update=0
        for epoch in range(max_epoch):
            print(f'Epoch {epoch+1}:')
            total_batch_num=int(np.ceil(num_train/batch_size))
            for batch_id in tqdm(range(total_batch_num)):
                batch_idxs=idxs[:,batch_id*batch_size:(batch_id+1)*batch_size]
                batch_input=train_inputs[batch_idxs]
                batch_target=train_targets[batch_idxs]

                input_tensor=torch.tensor(batch_input,dtype=torch.float32,device=self.device)
                input_tensor=self.scaler.transform(input_tensor)
                target_tensor=torch.tensor(batch_target,dtype=torch.float32,device=self.device)
                # [obs reward] [obs action]

                self.nets.optimizer.zero_grad()
                output_mean_tensor,log_var_tensor=self.nets(input_tensor,ret_log_var=True)
                loss,mse_loss=self.nets.compute_loss(output_mean_tensor,log_var_tensor,target_tensor)
                loss.backward()
                self.nets.optimizer.step()
                grad_update+=1
                # if (batch_id+1)%100==0:
                #     print(f'{epoch+1} Epoch {loss.item():.5f}')


            idxs=np.argsort(np.random.rand(E,num_train),axis=-1)

            train_mse_loss=self.measure_mse_loss(train_inputs,train_targets)

            msg=''
            for i in range(E):
                msg+=f'T{i}: {train_mse_loss[i].item():.6f}  '
            print(msg)

            holdout_mse_loss=self.measure_mse_loss(holdout_inputs,holdout_targets)

            msg=''
            for i in range(E):
                msg+=f'V{i}: {holdout_mse_loss[i].item():.6f}  '
            print(msg,'\n')

    @torch.no_grad()
    def measure_mse_loss(self,input,target):
        assert input.ndim==2 and target.ndim==2
        batch_size=1024
        data_num=input.shape[0]
        E=self.nets.ensemble_size

        batch_num=int(np.ceil(data_num/batch_size))
        total_mse_loss=0.0
        for i in range(batch_num):
            batch_data=np.tile(input[i*batch_size:(i+1)*batch_size][None],[E,1,1])
            target_data=np.tile(target[i*batch_size:(i+1)*batch_size][None],[E,1,1])
            input_tensor=torch.tensor(batch_data,dtype=torch.float32,device=self.device)
            input_tensor=self.scaler.transform(input_tensor)
            target_tensor=torch.tensor(target_data,dtype=torch.float32,device=self.device)

            B=input_tensor.shape[1]

            mean,logvar=self.nets(input_tensor,ret_log_var=True)
            _,mse_loss=self.nets.compute_loss(mean,logvar,target_tensor)
            total_mse_loss+=B*mse_loss
        return total_mse_loss/data_num

    # @torch.no_grad()        
    # def measure_holdout_loss(self,holdout_input,holdout_target):
    #     assert holdout_input.ndim==3 and holdout_target.ndim==3
    #     batch_size=512
    #     holdout_num=holdout_input.shape[1]

    #     batch_num=int(np.ceil(holdout_num/batch_size))
    #     total_mse_loss=0.0
    #     for i in range(batch_num):
    #         input_tensor=torch.tensor(holdout_input[:,i*batch_size:(i+1)*batch_size],dtype=torch.float32,device=self.device)
    #         input_tensor=self.scaler.transform(input_tensor)
    #         target_tensor=torch.tensor(holdout_target[:,i*batch_size:(i+1)*batch_size],dtype=torch.float32,device=self.device)
            
    #         B=input_tensor.shape[1]
            
    #         mean,logvar=self.nets(input_tensor,ret_log_var=True)
    #         _,mse_loss=self.nets.compute_loss(mean,logvar,target_tensor)
    #         total_mse_loss+=B*mse_loss
    #     return total_mse_loss/holdout_num

        

if __name__=='__main__':
    # B,O,A=20000,4,2
    # obs=np.random.randn(B,O)
    # obs2=np.random.randn(B,O)
    # act=np.random.randn(B,A)
    # rew=np.random.randn(B)
    # dynamic_model=Ensemble_Dynamic_Model(O,A)
    # dynamic_model.train(obs,act,obs2,rew)
    
    import d4rl
    import gym
    from offlinespinup.utils.replay_buffer import ReplayBuffer

    env=gym.make("hopper-medium-replay-v0")
    dataset=d4rl.qlearning_dataset(env)
    act_dim=env.action_space.shape[0]
    obs_dim=env.observation_space.shape[0]

    replay_buffer=ReplayBuffer(obs_dim,act_dim,int(1e6))
    replay_buffer.load_dataset(dataset)

    dynamic_model=Ensemble_Dynamic_Model(obs_dim,act_dim)

    dynamic_model.train(replay_buffer.obs_buf,replay_buffer.act_buf,replay_buffer.obs2_buf,replay_buffer.rew_buf,max_epoch=100)

        


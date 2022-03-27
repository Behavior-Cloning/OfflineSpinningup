import bs4
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
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
        if data.ndim==2:
            return (data - self.mu) / self.std
        else:
            return (data-self.mu.unsqueeze(0))/self.std.unsqueeze(0)

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
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.network_size=network_size
        self.elite_size=elite_size
        self.deterministic=deterministic
        self.to(device)

    def train(self,obs,act,obs2,rew,batch_size=32,holdout_ratio=0.1,max_holdout_num=20000,max_epoch=100):
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

        idxs=np.argsort(np.random.rand(E,num_train),axis=-1)
        grad_update=0
        scheduler=MultiStepLR(self.nets.optimizer,[50,100,200],0.3)
        for epoch in range(max_epoch):
            if epoch==0:
                print('BEFORE TRAINING:')
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
            # print(train_inputs[0:5])
            # print(train_targets[0:5])
            print(f'Epoch {epoch+1}:')
            total_batch_num=int(np.ceil(num_train/batch_size))
            # for batch_id in tqdm(range(total_batch_num)):
            for batch_id in range(total_batch_num):
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

            scheduler.step()

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

    @torch.no_grad()
    def batch_predict(self,obs,action):
        obs_tensor=torch.clone(obs)
        action_tensor=torch.clone(action)
        assert obs_tensor.dim()==3 and action_tensor.dim()==3 and obs_tensor.shape[0]==action_tensor.shape[0]
        input_tensor=torch.cat([obs_tensor,action_tensor],dim=-1)
        input_tensor=self.scaler.transform(input_tensor)
        outputs, _ = self.nets(input_tensor, ret_log_var=False)
        # outputs = outputs.mean(0)
        #reward obs (E,B,) (E,B,O)
        return outputs[..., -1], outputs[..., :-1] + obs_tensor.unsqueeze(0)

    def predict(self,obs,action):
        if not isinstance(obs,torch.Tensor):
            obs_tensor=torch.tensor(obs,dtype=torch.float32,device=self.device)
        else:
            obs_tensor=torch.clone(obs).device(self.device)

        if not isinstance(action,torch.Tensor):
            action_tensor=torch.tensor(action,dtype=torch.float32,device=self.device)
        else:
            action_tensor=torch.clone(action).device(self.device)
        
        if obs_tensor.ndim==2:
            obs_tensor=obs_tensor.unsqueeze_(0).repeat(self.network_size,1,1)
        if action_tensor.ndim==2:
            action_tensor=action_tensor.unsqueeze_(0).repeat(self.network_size,1,1)
        
        assert action_tensor.shape[0]==self.network_size and obs_tensor.shape[0]==self.network_size

        input_tensor=torch.cat([obs_tensor,action_tensor],dim=-1)
        input_tensor=self.scaler.transform(input_tensor)

        mean_tensor,var_tensor=self.nets(input_tensor,ret_log_var=False)
        #(E,B,O+1) (E,B,O+1)   deterministic: (E,B,O+1), None

        if self.deterministic:
            delta_tensor=mean_tensor[...,:-1]
            reward_tensor=mean_tensor[...,-1]
        else:
            std_tensor=torch.sqrt(var_tensor)
            sample_tensor=mean_tensor+torch.randn(tuple(mean_tensor.shape)).to(mean_tensor)*std_tensor
            delta_tensor=sample_tensor[...,:-1]
            reward_tensor=sample_tensor[...,-1]
        # obs(E,B,O) reward(E,B) torch cuda tensor
        return delta_tensor+obs_tensor,reward_tensor



if __name__=='__main__':
    # B,O,A=20000,4,2
    # obs=np.random.randn(B,O)
    # obs2=obs+1.0
    # act=np.random.randn(B,A)
    # rew=np.ones(B)*0.2
    # dynamic_model=Ensemble_Dynamic_Model(O,A)
    # dynamic_model.train(obs,act,obs2,rew,batch_size=512,max_epoch=20)
    # print(obs[:5])
    # pred_obs2,reward2=dynamic_model.predict(obs,act)
    # print(pred_obs2[:5])
    # print(reward2[:5])



    import argparse
    import d4rl
    import gym
    from offlinespinup.utils.replay_buffer import ReplayBuffer
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=256)
    args=parser.parse_args()
    seed=args.seed
    bs=args.batch_size

    np.random.seed(seed)
    torch.manual_seed(seed)
    env=gym.make("hopper-medium-v0")
    env.seed(seed)
    dataset=d4rl.qlearning_dataset(env)
    act_dim=env.action_space.shape[0]
    obs_dim=env.observation_space.shape[0]

    replay_buffer=ReplayBuffer(obs_dim,act_dim,int(1e6))
    replay_buffer.load_dataset(dataset)

    dynamic_model=Ensemble_Dynamic_Model(obs_dim,act_dim,lr=1e-3)

    dynamic_model.train(replay_buffer.obs_buf,replay_buffer.act_buf,replay_buffer.obs2_buf,replay_buffer.rew_buf,
        max_epoch=300,batch_size=bs)

        


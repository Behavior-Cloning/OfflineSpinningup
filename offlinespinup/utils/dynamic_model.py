import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import sys
import os
import copy

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
    def __init__(self,obs_dim,act_dim,rew_dim=1,ensemble_size=7,hidden_dim=200):
        super().__init__()
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.rew_dim=rew_dim
        self.ensemble_size=ensemble_size
        self.hidden_dim=hidden_dim
        self.out_dim=2*(obs_dim+rew_dim)

        self.nn1=Ensemble_Linear(obs_dim+act_dim,hidden_dim,ensemble_size,weight_decay=0.000025,init_method=truncated_normal_init)
        self.nn2=Ensemble_Linear(hidden_dim,hidden_dim,ensemble_size,weight_decay=0.00005,init_method=truncated_normal_init)
        self.nn3=Ensemble_Linear(hidden_dim,hidden_dim,ensemble_size,weight_decay=0.000075,init_method=truncated_normal_init)
        self.nn4=Ensemble_Linear(hidden_dim, hidden_dim, ensemble_size, weight_decay=0.000075,init_method=truncated_normal_init)
        self.nn5=Ensemble_Linear(hidden_dim,self.out_dim,ensemble_size,weight_decay=0.0001,init_method=truncated_normal_init)
        self.max_logvar = nn.Parameter(torch.ones(1, obs_dim+rew_dim).float() * 0.5, requires_grad=False)
        self.min_logvar = nn.Parameter(torch.ones(1, obs_dim+rew_dim).float() * -10, requires_grad=False)
        self.swish=Swish()
        
    def forward(self,x,ret_log_var=False):
        nn1_output=self.swish(self.nn1(x))
        nn2_output=self.swish(self.nn2(nn1_output))
        nn3_output=self.swish(self.nn3(nn2_output))
        nn4_output=self.swish(self.nn4(nn3_output))
        nn5_output=self.nn5(nn4_output)

        mean,logvar=nn5_output.chunk(2,dim=-1)
        logvar=logvar-F.softplus(logvar-self.max_logvar)
        logvar=logvar+F.softplus(self.min_logvar-logvar)

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
    def __init__(self,obs_dim,act_dim,domain,reward_dim=1,network_size=7,elite_size=5,hidden_dim=200,
    lr=1e-3,deterministic=False,device='cuda:0'):
        super().__init__()
        self.nets=Ensemble_Model(obs_dim,act_dim,reward_dim,network_size,hidden_dim)
        self.scaler=StandardScaler(obs_dim+act_dim)
        self.device=device
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.network_size=network_size
        self.elite_size=elite_size
        self.deterministic=deterministic
        self.to(device)

        self._cached_state_dict=copy.deepcopy(self.nets.state_dict())
        self._holdout_losses=None

        param_group=[]
        for m in self.nets.children():
            if isinstance(m,Ensemble_Linear):
                param_group+=[{'params':m.weight,'weight_decay':m.weight_decay},
                                {'params':m.bias}]

        self.optimizer=Adam(param_group,lr)

        self.elite_inds=None

        self.domain=domain# list for meta learning, str for single domain
        self.trained_flag=False

    def _save_states(self):
        self._cached_state_dict=copy.deepcopy(self.nets.state_dict())

    def _save_state(self,id):
        state_dict=self.nets.state_dict()
        for k,v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                self._cached_state_dict[k].data[id]=copy.deepcopy(v.data[id])

    def _save_best(self,holdout_loss):
        if self._holdout_losses is None:
            self._holdout_losses=copy.deepcopy(holdout_loss)
        else:
            for model_id in range(self.network_size):
                if holdout_loss[model_id]<self._holdout_losses[model_id]:
                    self._save_state(model_id)
                    self._holdout_losses[model_id]=holdout_loss[model_id]

    def _end_train(self):
        """load the best parameters as the final model"""

        self.nets.load_state_dict(self._cached_state_dict)
        self.trained_flag=True
        self.elite_inds=(np.argsort(self._holdout_losses.detach().cpu())[:self.elite_size]).tolist()
        print(f"Using {self.elite_size}/{self.network_size} models: {self.elite_inds}")

    def _eval(self,train_inputs,train_targets,holdout_inputs,holdout_targets):
        E=self.nets.ensemble_size
        train_mse_loss=self._measure_mse_loss(train_inputs,train_targets)
        msg=''
        for i in range(E):
            msg+=f'T{i}: {train_mse_loss[i].item():.6f}  '
        print(msg)
        
        holdout_mse_loss=self._measure_mse_loss(holdout_inputs,holdout_targets)
        msg=''
        for i in range(E):
            msg+=f'V{i}: {holdout_mse_loss[i].item():.6f}  '
        print(msg,'\n')

        self._save_best(holdout_mse_loss)

        return train_mse_loss,holdout_mse_loss

    def train(self,obs,act,obs2,rew,batch_size=32,holdout_ratio=0.1,max_holdout_num=10000,max_epoch=100):
        """numpy ndarray input"""
        for params in self.optimizer.param_groups:
            params['lr']=1e-3

        self._holdout_losses=None
        self._save_states()

        inputs=np.concatenate((obs,act),axis=-1)
        targets=np.concatenate((obs2-obs,rew[:,None] if rew.ndim==1 else rew),axis=-1)
        self.scaler.fit(torch.tensor(inputs,dtype=torch.float32,device=self.device))

        total_num=obs.shape[0]

        num_holdout=int(min(total_num*holdout_ratio,max_holdout_num))
        num_holdout=1 if num_holdout<=0 else num_holdout
        num_train=total_num-num_holdout

        if num_train==0:
            raise("There are no data in training set!")

        E=self.nets.ensemble_size
        O=obs.shape[-1]

        permutation = np.random.permutation(total_num)
        train_inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        train_targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        print(f"Split Dataset: |validation set|={num_holdout}, |training set|={num_train}\n")

        idxs=np.argsort(np.random.rand(E,num_train),axis=-1)# shuffle inputs, no bagging here, which is different with the official code
        grad_update=0
        scheduler=MultiStepLR(self.optimizer,[50,100,200],0.3)# lr decay, added

        TrainLoss,ValLoss=[],[]

        for epoch in range(max_epoch):
            if epoch==0:
                print('BEFORE TRAINING:')
                self._eval(train_inputs,train_targets,holdout_inputs,holdout_targets)

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

                self.optimizer.zero_grad()
                output_mean_tensor,log_var_tensor=self.nets(input_tensor,ret_log_var=True)
                loss,mse_loss=self.nets.compute_loss(output_mean_tensor,log_var_tensor,target_tensor)
                loss.backward()
                self.optimizer.step()
                grad_update+=1

            idxs=np.argsort(np.random.rand(E,num_train),axis=-1) # shuffle input after one epoch

            train_mse_loss,holdout_mse_loss=self._eval(train_inputs,train_targets,holdout_inputs,holdout_targets)
        
            TrainLoss.append(train_mse_loss.detach().cpu().numpy())
            ValLoss.append(holdout_mse_loss.detach().cpu().numpy())

            scheduler.step()# per epoch decay, note that this will have different effect to the distinct batch size
        
        self._end_train()
        print('Training end! Model final performance: ')
        self._eval(train_inputs,train_targets,holdout_inputs,holdout_targets)

    @torch.no_grad()
    def _measure_mse_loss(self,input,target):
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
    def predict(self,obs,action,penalty_coef=0.0,uncertainty_type='disagreement'):
        if not self.trained_flag:
            print("\nWarning: the model has not been trained yet!\n")

        if not isinstance(obs,torch.Tensor):
            obs_tensor=torch.tensor(obs,dtype=torch.float32,device=self.device)
        else:
            obs_tensor=torch.clone(obs).to(self.device)

        if not isinstance(action,torch.Tensor):
            action_tensor=torch.tensor(action,dtype=torch.float32,device=self.device)
        else:
            action_tensor=torch.clone(action).to(self.device)
        
        if obs_tensor.ndim==2:
            obs_tensor=obs_tensor.unsqueeze_(0).repeat(self.network_size,1,1)
        if action_tensor.ndim==2:
            action_tensor=action_tensor.unsqueeze_(0).repeat(self.network_size,1,1)
        
        assert action_tensor.shape[0]==self.network_size and obs_tensor.shape[0]==self.network_size

        input_tensor=torch.cat([obs_tensor,action_tensor],dim=-1)
        input_tensor=self.scaler.transform(input_tensor)

        mean_tensor,var_tensor=self.nets(input_tensor,ret_log_var=False)
        #(E,B,O+1) (E,B,O+1)   deterministic: (E,B,O+1), None
        delta_tensor=mean_tensor[...,:-1]
        
        reward_tensor=mean_tensor[...,-1]
        obs2_tensor=delta_tensor+obs_tensor
        obs2_std_tensor=torch.sqrt(var_tensor)[...,:-1]
        reward_std_tensor=torch.sqrt(var_tensor)[...,-1]
        
        uncertainty=0.0
        # if penalty_coef !=0.0:
        obs2_tensor_mode=torch.mean(obs2_tensor,dim=0,keepdim=True)
        diff=obs2_tensor-obs2_tensor_mode
        disagreement_uncertainty=torch.max(torch.norm(diff,dim=-1),dim=0,keepdim=True)[0]
        aleatoric_uncertainty=torch.max(torch.norm(obs2_std_tensor,dim=-1),dim=0,keepdim=True)[0]
        uncertainty=disagreement_uncertainty if uncertainty_type=='disagreement' else aleatoric_uncertainty
        assert uncertainty.dim()==2 
        
        if not self.deterministic:
            obs2_tensor+=torch.randn(tuple(obs2_tensor.shape)).to(obs2_tensor)*obs2_std_tensor
            reward_tensor+=torch.randn(tuple(reward_tensor.shape)).to(reward_tensor)*reward_std_tensor
        #(E,B,O) (E,B)
        info={"ModelReward":reward_tensor.detach().cpu().numpy(),
                "ModelUncertainty":uncertainty.detach().cpu().numpy()}
        return obs2_tensor,reward_tensor-penalty_coef*uncertainty,info
    

if __name__=='__main__':
    # B,O,A=20000,4,2
    # np.random.seed(10)
    # obs=np.random.randn(B,O)
    # obs2=obs**2
    # act=np.random.randn(B,A)
    # rew=np.ones(B)*0.2
    # dynamic_model=Ensemble_Dynamic_Model(O,A,'toy')
    # # dynamic_model=torch.load('toy_model.pth')
    # dynamic_model.train(obs,act,obs2,rew,batch_size=1024,max_epoch=5)
    # # print(obs[:3])

    # task='halfcheetah-medium-replay-v0'
    # prefix='resource/'
    # seed=20
    # file_path=prefix+task.replace('-','_')+f'/model_{seed}.pth'

    # torch.save(dynamic_model,file_path)

    # dynamic_model=None
    # dynamic_model=torch.load(file_path)
    # print(dynamic_model.elite_inds)
    # print(dynamic_model.act_dim)
    # print(dynamic_model.trained_flag)
    # print(dynamic_model._holdout_losses)
    # print(dynamic_model.elite_inds)


    import argparse
    import d4rl
    import gym
    from offlinespinup.utils.replay_buffer import ReplayBuffer
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--task",type=str,default="halfcheetah-medium-replay-v0")
    args=parser.parse_args()
    seed=args.seed
    bs=args.batch_size

    task=args.task

    np.random.seed(seed)
    torch.manual_seed(seed)
    env=gym.make(task)
    env.seed(seed)
    dataset=d4rl.qlearning_dataset(env)
    act_dim=env.action_space.shape[0]
    obs_dim=env.observation_space.shape[0]

    replay_buffer=ReplayBuffer(obs_dim,act_dim,int(1e6))
    replay_buffer.load_dataset(dataset)

    dynamic_model=Ensemble_Dynamic_Model(obs_dim,act_dim,task)

    dynamic_model.train(replay_buffer.obs_buf,replay_buffer.act_buf,replay_buffer.obs2_buf,replay_buffer.rew_buf,
        max_epoch=500,batch_size=bs)

    prefix='resource/'
    file_path=prefix+task.replace('-','_')+f'/model_{seed}.pth'
    torch.save(dynamic_model,file_path)

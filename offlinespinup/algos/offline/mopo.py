import d4rl
import gym
import torch
import numpy as np

import os
import sys

base_dir=os.path.abspath(__file__)
dir=os.path.abspath(__file__)
for ii in range(4):
    dir=os.path.dirname(dir)
sys.path.append(dir)

from torch.optim import Adam
from offlinespinup.utils.sac_policy import SACpolicy
from offlinespinup.utils.replay_buffer import ReplayBuffer
from offlinespinup.utils.logger import EasyLogger
from offlinespinup.utils.dynamic_model import *
from offlinespinup.utils.terminal_check import is_terminal


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


        
def mopo(env,test_env,task,critic_ensemble_size=2,seed=0,epochs=1000,steps_per_epoch=1000,device='cuda:0',
replay_buffer_size=int(1e6),model_load_path=None,model_retain_epochs=5,rollout_batch_size=50000,rollout_length=5,rollout_freq=1000,
real_ratio=0.05,penalty_coef=1.0,max_ep_len=1000,batch_size=256,num_test_episodes=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    dataset=d4rl.qlearning_dataset(env)
    act_dim=env.action_space.shape[0]
    obs_dim=env.observation_space.shape[0]

    replay_buffer=ReplayBuffer(obs_dim,act_dim,replay_buffer_size)
    replay_buffer.load_dataset(dataset)

    N=10
    c_kwargs=dict(ensemble_size=critic_ensemble_size,hidden_sizes=[256,256])
    sac_agent=SACpolicy(obs_dim,act_dim,c_kwargs=c_kwargs,num_samples=N)
    sac_agent.to_device(device)

    a_optimizer=Adam(sac_agent.actor.parameters(),lr=3e-4)
    c_optimizer=Adam(sac_agent.critic.parameters(),lr=3e-4)

    logger=EasyLogger()
    var_counts = tuple(count_vars(module) for module in [sac_agent.actor, sac_agent.critic])
    logger.log('\nNumber of parameters: \t pi: %d, \t qs: %d\n'%var_counts)
    logger.log('Actor Net:\n',sac_agent.actor.net)
    logger.log('\nCritic Net:\n',sac_agent.critic.net,'\n')

    if model_load_path==None:
        raise("model load path is None!")
    model=torch.load(model_load_path)

    model_buffer=ReplayBuffer(obs_dim,act_dim,model_retain_epochs*rollout_batch_size*rollout_length)
    assert isinstance(model,Ensemble_Dynamic_Model)

    @torch.no_grad()
    def rollout(obs0):
        # print(obs0)
        obs_tensor=torch.clone(obs0).to(model.device)
        rollout_num=0.0
        
        for _ in range(rollout_length):
            assert obs_tensor.dim()==2
            B=obs_tensor.shape[0]
            rollout_num+=B
            action,_=sac_agent.actor(obs_tensor,num_samples=1)#(N,B,A)
            action=action.squeeze(0)#(B,A)
            assert action.shape[0]==B
            obs2_tensor,reward_tensor,model_info=model.predict(obs_tensor,action,penalty_coef=penalty_coef)
            #(E,B,O) (E,B)

            obs2_tensor=torch.mean(obs2_tensor,dim=0)
            reward_tensor=torch.mean(reward_tensor,dim=0)
            assert obs2_tensor.dim()==2 and reward_tensor.dim()==1
            obs_np,act_np,obs2_np,rew_np=obs_tensor.detach().cpu().numpy(),action.detach().cpu().numpy(),obs2_tensor.detach().cpu().numpy(),reward_tensor.detach().cpu().numpy()
            terminal_flag=is_terminal(obs_np,act_np,obs2_np,task)
            done_np=np.array(terminal_flag,dtype=np.float32)

            model_buffer.store_batch(obs_np,act_np,rew_np,obs2_np,done_np) 
            # nonterm_mask = (~torch.tensor(terminal_flag,dtype=torch.bool)).to(obs2_tensor).flatten()
            nonterm_mask = (~torch.tensor(terminal_flag,dtype=torch.bool)).flatten()
            if nonterm_mask.sum() == 0:
                break

            obs_tensor = obs2_tensor[nonterm_mask]

        return rollout_num
    
    @torch.no_grad()
    def model_rollout():
        single_batch_size=1024
        total_rollout_num=0.0
        print('\nStart rollout:')
        while total_rollout_num<rollout_batch_size:
            data=replay_buffer.sample_batch(single_batch_size)
            total_rollout_num+=rollout(data['obs'])
        logger.log(f'{int(total_rollout_num)} transitions are added into model buffer(total {model_buffer.size})!\n')
    
    def syn_train_data():
        real_sample_size=int(batch_size*real_ratio)
        fake_sample_size=batch_size-real_sample_size
        real_batch=replay_buffer.sample_batch(real_sample_size,device)
        fake_batch=model_buffer.sample_batch(fake_sample_size,device)
        batch={
            'obs':torch.cat([real_batch['obs'],fake_batch['obs']],dim=0),
            'obs2':torch.cat([real_batch['obs2'],fake_batch['obs2']],dim=0),
            'rew':torch.cat([real_batch['rew'],fake_batch['rew']],dim=0),
            'done':torch.cat([real_batch['done'],fake_batch['done']],dim=0),
            'act':torch.cat([real_batch['act'],fake_batch['act']],dim=0)
        }
        return batch

    def update(data,cur_steps):
        c_optimizer.zero_grad()
        loss_q_sac,q_info=sac_agent.compute_q_loss(data)
        loss_q=loss_q_sac
        loss_q.backward()
        c_optimizer.step()
        logger.store(QLoss=loss_q.item(),**q_info)

        for p in sac_agent.critic.parameters():
            p.requires_grad=False

        a_optimizer.zero_grad()
        loss_pi,pi_info=sac_agent.compute_pi_loss(data)
        loss_pi.backward()
        a_optimizer.step()
        logger.store(PiLoss=loss_pi.item(),**pi_info)

        for p in sac_agent.critic.parameters():
            p.requires_grad=True
        
        sac_agent.syn_weight()

    def test_agent():
            for i in range(num_test_episodes):
                ep_ret,o,ep_len,d=0.0,test_env.reset(),0,False
                while not(d or ep_len>=max_ep_len):
                    o,r,d,_=test_env.step(sac_agent.get_action(o,deterministic=True))
                    ep_ret+=r
                    ep_len+=1
                logger.store(Performance=ep_ret,NormalizedScore=env.get_normalized_score(ep_ret))

    total_steps=int(epochs*steps_per_epoch)

    for t in range(total_steps):
        if t%rollout_freq==0:
            model_rollout()
        
        batch = syn_train_data()
        update(data=batch,cur_steps=t)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            test_agent()
            logger.log(f'\nEpoch {epoch}/{epochs}, Steps [{t+1}/{total_steps}] {t/total_steps*100:.2f}%')
            logger.dump_records()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default="walker2d-medium-replay-v0")
    parser.add_argument('--rollout_length',type=int,default=1)
    parser.add_argument('--penalty_coef',type=float,default=1.0)
    args=parser.parse_args()

    task=args.task
    model_load_path='resource/'+task.replace('-','_')+'/model_0.pth'

    env=gym.make(task)
    test_env=gym.make(task)
    mopo(env,test_env,task,model_load_path=model_load_path,rollout_length=args.rollout_length,penalty_coef=args.penalty_coef)
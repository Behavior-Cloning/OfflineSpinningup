import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import os
import sys

base_dir=os.path.abspath(__file__)
dir=os.path.abspath(__file__)
for ii in range(4):
    dir=os.path.dirname(dir)
sys.path.append(dir)

from offlinespinup.utils.sac_policy import SACpolicy
from offlinespinup.utils.replay_buffer import ReplayBuffer
from offlinespinup.utils.logger import EasyLogger


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def sac(env,test_env,seed=0,epochs=50,steps_per_epoch=4000,start_steps=10000,
update_every=50,update_after=1000,device='cuda:0',replay_buffer_size=int(1e6),
max_ep_len=1000,batch_size=100,num_test_episodes=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)
    test_env.action_space.seed(seed)

    obs_dim=env.observation_space.shape[0]
    act_dim=env.action_space.shape[0]

    act_limit=env.action_space.high[0]

    # print(obs_dim,act_dim)
    sac_agent=SACpolicy(obs_dim,act_dim)
    sac_agent.to_device(device)
    a_optimizer=Adam(sac_agent.actor.parameters(),lr=1e-3)
    c_optimizer=Adam(sac_agent.critic.parameters(),lr=1e-3)

    replay_buffer=ReplayBuffer(obs_dim,act_dim,replay_buffer_size)
    
    logger=EasyLogger()

    var_counts = tuple(count_vars(module) for module in [sac_agent.actor, sac_agent.critic])
    logger.log('\nNumber of parameters: \t pi: %d, \t qs: %d\n'%var_counts)
    logger.log('Actor Net:\n',sac_agent.actor.net)
    logger.log('\nCritic Net:\n',sac_agent.critic.net,'\n')

    def update(data):
        c_optimizer.zero_grad()
        loss_q,q_info=sac_agent.compute_q_loss(data)
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
            logger.store(Performance=ep_ret)
            

    total_steps=int(epochs*steps_per_epoch)
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):

        if t > start_steps:
            a = sac_agent.get_action(o,deterministic=False)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len==max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size,device)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            test_agent()
            logger.log(f'\nEpoch {epoch}/{epochs}, Steps [{t+1}/{total_steps}] {t/total_steps*100:.2f}%')
            logger.dump_records()



if __name__=='__main__':
    env=gym.make('HalfCheetah-v3')
    test_env=gym.make('HalfCheetah-v3')
    sac(env,test_env)

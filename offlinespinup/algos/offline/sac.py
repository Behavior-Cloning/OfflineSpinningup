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


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def sac(env,test_env,critic_ensemble_size=2,seed=0,epochs=200,steps_per_epoch=5000,device='cuda:0',replay_buffer_size=int(1e6),max_ep_len=1000,batch_size=100,num_test_episodes=100):
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
        batch = replay_buffer.sample_batch(batch_size,device)
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
    parser.add_argument('--task',type=str,default="hopper-medium-replay-v0")
    parser.add_argument('--critic_ensemble_size',type=int,default=2)
    args=parser.parse_args()

    env=gym.make(args.task)
    test_env=gym.make(args.task)

    sac(env,test_env,critic_ensemble_size=args.critic_ensemble_size)
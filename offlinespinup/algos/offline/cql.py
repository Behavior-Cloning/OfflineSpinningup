import d4rl
import gym
import torch
import numpy as np

from torch.optim import Adam
from offlinespinup.utils.sac_policy import SACpolicy
from offlinespinup.utils.replay_buffer import ReplayBuffer
from offlinespinup.utils.logger import EasyLogger


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def cql(env,seed=0,epochs=200,steps_per_epoch=5000,start_steps=10000,device='cuda:0',replay_buffer_size=int(1e6),max_ep_len=1000,batch_size=100,num_test_episodes=100):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset=d4rl.qlearning_dataset(env)
    act_dim=env.action_space.shape[0]
    obs_dim=env.observation_space.shape[0]

    replaybuffer=ReplayBuffer(obs_dim,act_dim,replay_buffer_size)
    replaybuffer.load_dataset(dataset)


    N=10
    sac_agent=SACpolicy(obs_dim,act_dim,num_samples=N)
    sac_agent.to_device(device)
    a_optimizer=Adam(sac_agent.actor.parameters(),lr=1e-3)
    c_optimizer=Adam(sac_agent.critic.parameters(),lr=1e-3)


    logger=EasyLogger()
    var_counts = tuple(count_vars(module) for module in [sac_agent.actor, sac_agent.critic])
    logger.log('\nNumber of parameters: \t pi: %d, \t qs: %d\n'%var_counts)
    logger.log('Actor Net:\n',sac_agent.actor.net)
    logger.log('\nCritic Net:\n',sac_agent.critic.net,'\n')

    def conservative_loss(data):
        o,a,o2=data['obs'],data['act'],data['obs2']
        B,O=tuple(o.shape)
        A=act_dim

        rand_actions=torch.rand((N,A),device=device,dtype=torch.float32).uniform_(-1,1)
        


    def update(data):
        


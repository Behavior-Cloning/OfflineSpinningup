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

def cql(env,test_env,seed=0,epochs=200,steps_per_epoch=5000,start_steps=10000,device='cuda:0',replay_buffer_size=int(1e6),max_ep_len=1000,batch_size=100,num_test_episodes=100):
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
    conservative_weight=5.0
    sac_agent=SACpolicy(obs_dim,act_dim,num_samples=N)
    sac_agent.to_device(device)
    a_optimizer=Adam(sac_agent.actor.parameters(),lr=1e-3)
    c_optimizer=Adam(sac_agent.critic.parameters(),lr=1e-3)

    logger=EasyLogger()
    var_counts = tuple(count_vars(module) for module in [sac_agent.actor, sac_agent.critic])
    logger.log('\nNumber of parameters: \t pi: %d, \t qs: %d\n'%var_counts)
    logger.log('Actor Net:\n',sac_agent.actor.net)
    logger.log('\nCritic Net:\n',sac_agent.critic.net,'\n')

    def compute_bc_loss(data):
        """simple L2 norm version"""
        o,a=data['obs'],data['act']
        pred_actions,logpi=sac_agent.actor(o,deterministic=False,with_logp_pi=True,num_samples=N)
        pi_info=dict(LogPi=logpi.detach().cpu().numpy())
        bc_loss=torch.mean((pred_actions-a.unsqueeze(0))**2)
        return bc_loss,pi_info

    def compute_conservative_loss(data):
        o,a,o2=data['obs'],data['act'],data['obs2']
        B,O=tuple(o.shape)
        A=act_dim

        with torch.no_grad():
            rand_actions=torch.rand((N,B,A),device=device,dtype=torch.float32).uniform_(-1,1)
            rand_actions_logp_pi=np.log(0.5**act_dim)
            #(N,B,A) scale

            cur_actions,cur_actions_logp_pi=sac_agent.actor(o,deterministic=False,with_logp_pi=True,num_samples=N)
            #(N,B,A) (N,B)

            next_actions,next_actions_logp_pi=sac_agent.actor(o2,deterministic=False,with_logp_pi=True,num_samples=N)
            #(N,B,A) (N,B)

        q_random_actions=sac_agent.critic(o,rand_actions)
        q_cur_actions=sac_agent.critic(o,cur_actions)
        q_next_actions=sac_agent.critic(o,next_actions)
        #(N,E,B)

        cat_is_q=torch.cat([q_random_actions-rand_actions_logp_pi,q_cur_actions-cur_actions_logp_pi.unsqueeze(1),\
                    q_next_actions-next_actions_logp_pi.unsqueeze(1)],dim=0)
        assert tuple(cat_is_q.shape)==(3*N,sac_agent.critic.ensemble_size,B)

        conservative_loss=torch.logsumexp(cat_is_q,dim=0).mean()-sac_agent.critic(o,a).mean()
        return conservative_weight*conservative_loss

    def update(data,cur_steps):
        c_optimizer.zero_grad()
        loss_q_sac,q_info=sac_agent.compute_q_loss(data)
        conservative_loss=compute_conservative_loss(data)
        loss_q=loss_q_sac+conservative_loss
        loss_q.backward()
        c_optimizer.step()
        logger.store(QLoss=loss_q.item(),OriginSACQLoss=loss_q_sac.item(),ConservativeLoss=conservative_loss.item(),**q_info)

        for p in sac_agent.critic.parameters():
            p.requires_grad=False

        a_optimizer.zero_grad()
        if cur_steps<start_steps:
            loss_pi,pi_info=compute_bc_loss(data)
        else:
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
    parser.add_argument("--env_id",'-e',type=str,default='hopper-medium-replay-v0')
    args=parser.parse_args()

    env=gym.make(args.env_id)
    test_env=gym.make(args.env_id)
    cql(env,test_env)
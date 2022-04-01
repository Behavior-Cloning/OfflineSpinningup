import d4rl
import torch
import gym

task='walker2d-medium-replay-v0'
env=gym.make(task)
dataset=d4rl.qlearning_dataset(env)
torch.save(dataset,task+'_dataset')

print(env.action_space.shape)
print(env.observation_space.shape)
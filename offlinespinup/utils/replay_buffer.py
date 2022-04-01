from matplotlib.pyplot import isinteractive
import numpy as np
import torch

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32,target_deivce='cuda:0'):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32,device=target_deivce) for k,v in batch.items()}

    def load_dataset(self,dataset):
        self.obs_buf = np.array(dataset["observations"],dtype=np.float32)
        self.obs2_buf = np.array(dataset["next_observations"],dtype=np.float32)
        self.act_buf = np.array(dataset["actions"],dtype=np.float32)
        self.rew_buf = np.array(dataset["rewards"],dtype=np.float32).reshape(-1)
        self.done_buf = np.array(dataset["terminals"],dtype=np.float32).reshape(-1)

        self.ptr, self.size, self.max_size = len(self.obs_buf), len(self.obs_buf), len(self.obs_buf)
        print(f"\nSuccessfully load the static offine dataset(containing {self.size} transitions)!\n")

    def store_batch(self,obs_b,act_b,rew_b,obs2_b,done_b):
        assert isinstance(obs_b,np.ndarray) and isinstance(obs2_b,np.ndarray)\
            and isinstance(act_b,np.ndarray) and isinstance(rew_b,np.ndarray)\
                and isinstance(done_b,np.ndarray)
        assert obs_b.shape[0]==obs2_b.shape[0]==done_b.shape[0]==rew_b.shape[0]==done_b.shape[0]

        if rew_b.ndim==2:
            rew_b=rew_b.copy().reshape(-1)
        if done_b.ndim==2:
            done_b=done_b.copy().reshape(-1)
        
        B=obs_b.shape[0]
        batch_size = len(obs_b)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.obs_buf[begin:end] = np.array(obs_b[:first_add_size])
            self.obs2_buf[begin:end] = np.array(obs2_b[:first_add_size])
            self.act_buf[begin:end] = np.array(act_b[:first_add_size])
            self.rew_buf[begin:end] = np.array(rew_b[:first_add_size])
            self.done_buf[begin:end] = np.array(done_b[:first_add_size])

            begin = 0
            end = batch_size - first_add_size
            self.obs_buf[begin:end] = np.array(obs_b[first_add_size:]).copy()
            self.obs2_buf[begin:end] = np.array(obs2_b[first_add_size:]).copy()
            self.act_buf[begin:end] = np.array(act_b[first_add_size:]).copy()
            self.rew_buf[begin:end] = np.array(rew_b[first_add_size:]).copy()
            self.done_buf[begin:end] = np.array(done_b[first_add_size:]).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.obs_buf[begin:end] = np.array(obs_b).copy()
            self.obs2_buf[begin:end] = np.array(obs2_b).copy()
            self.act_buf[begin:end] = np.array(act_b).copy()
            self.rew_buf[begin:end] = np.array(rew_b).copy()
            self.done_buf[begin:end] = np.array(done_b).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)
        
        

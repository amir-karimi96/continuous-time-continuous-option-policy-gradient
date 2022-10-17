from traceback import print_tb
from env_wrapper import  D2C, Env_test, CT_pendulum, CT_pendulum_sparse,CT_mountain_car
import numpy as np
import torch 
import time


env = CT_mountain_car(dt = 0.007)
env.seed(0)


continuous_env = D2C(discrete_env= env, low_level_funciton= lambda x,y,z: x, rho = 1, precise=True)


print('reset: ', continuous_env.reset())

t0 = time.time()

s,r,done, info = continuous_env.step([1], 10)
undiscounted_rewards = info['rewards']
durations = info['durations']
print((np.array(undiscounted_rewards) * np.array(durations)).sum())
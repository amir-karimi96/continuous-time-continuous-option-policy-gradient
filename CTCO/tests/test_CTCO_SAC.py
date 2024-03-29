# actor critic continuous option continuous time 

import torch 
import numpy as np
from env_wrapper import  D2C, Env_test, CT_pendulum, CT_pendulum_sparse,CT_mountatin_car
import gym
import argparse
import yaml
import matplotlib.pyplot as plt
from dataset import Variable, Domain, RLDataset
from torch.utils.tensorboard import SummaryWriter
import time
from mpl_toolkits import mplot3d
from matplotlib import cm
import pickle
from agents import COCT_SAC

# Writer will output to ./runs/ directory by default
parser = argparse.ArgumentParser()
parser.add_argument('--ID',default=0,type=int, help='param ID ')
parser.add_argument('--config',default='0',type=str, help='config file name')
parser.add_argument('--result_path',default='0',type=str, help='result folder path')
parser.add_argument('--load_model', default=None,type=str, help='model')

args = parser.parse_args()
run_ID = args.ID
load_model = args.load_model
cfg_path = args.config
torch.manual_seed(run_ID * 1000) 
# print(cfg_path)
result_path = args.result_path
with open(cfg_path) as file:
    config = yaml.full_load(file)

param = config['param']
config_name = 'config_{}'.format(config['param_ID'])

       

def evaluate():
    pass

if __name__ == '__main__':


    writer = SummaryWriter(log_dir='{}/{}/result_{}'.format(result_path,config_name, run_ID))

    fig, axs = plt.subplots(3)

    t0 = time.time()
    


    # env = gym.make(param['env'])
    env = globals()[param['env']]()
    state_dim = len(env.observation_space.sample())
    action_dim = len(env.action_space.sample())
    config['action_high'] = env.action_space.high
    config['action_low'] = env.action_space.low
    config['state_dim'] = state_dim
    config['env_dt'] = env.dt
    config['writer'] = writer
    config['init_alpha'] = 1e-3
    config['log_alpha_requires_grad'] = True
    config['log_alpha_lr'] = 1e-4
    z_dim = param['z_dim']
    
    log_data = {'config_ID': config['param_ID'], 'config': param, 'data': None, }

    
    
    agent = COCT_SAC(config)
    if load_model is not None:
        #load
        agent.actor_network.load_state_dict(torch.load(load_model))
        

    num_ep = 3000
    continuous_env = D2C(discrete_env= env, low_level_funciton= lambda x,y,z: x, rho = agent.rho)

    
    t = 0.
    Returns = []
    for e in range(num_ep):
        s = continuous_env.reset()
        S = torch.tensor(s, dtype = torch.float32)
        
        undiscounted_rewards = []
        durations = []
        while True :

            # get actor net outputs based on current state and previous option
            predictions = agent.actor_network(S)
            
            
            
            # sample z
            z, _ = agent.get_action_z(predictions['z_mu'], predictions['z_sigma'])
            
            # sample duration 
            D, _ = agent.get_duration(predictions['D_mu'], predictions['D_sigma'])
            
            
            
            # sample movement omega
            omega = agent.z2omega(z) # TO DO add stochasticity

            d = D.detach().numpy()[0]
            
            sp,R,done,info = continuous_env.step(omega.detach().numpy(), d)
            # print('R: ', R )
            # print((np.array(info['rewards']) * np.array(info['durations'])).sum())
            if param['Duraiton_penalty']:
                R -= param['Duration_penalty_const']*d/D.detach().numpy()[0]
                # print(0.02*d/D.detach().numpy()[0])
            done_not_max = False
            if done:
                if info['TimeLimit.truncated']==False:
                    done_not_max = True
            
                
            agent.RB.notify(s=S.detach().numpy(), sp=sp, r=R, done_not_max=done_not_max, done=done,
                        z=z.detach().numpy(), 
                        D=D.detach().numpy(), d=d)
            
            undiscounted_rewards.extend(info['rewards'])
            durations.extend(info['durations'])
            # RB_list.append([s, sp, z, R, done, D, T, tau, agent.real_t])
            # print('R: ', R)
            agent.total_steps += 1
            if param['log_level'] == 2:
                writer.add_scalars('timings', {
                                                'D': D.detach().numpy(),
                                                'D_sigma': predictions['D_sigma'].detach().numpy(),
                                                
                                                }, agent.total_steps, walltime=agent.real_t)
                writer.add_scalars('action', {'z_mu': z,
                                                'z_sigma': predictions['z_sigma_target'][0],
                                                }, agent.total_steps, walltime=agent.real_t)
                writer.add_scalar('Reward', R, agent.total_steps)
            
                

            if agent.RB.real_size > 1 * config['param']['batch_size']:
                agent.update(num_epochs=len(info['durations']))

            if done:
                agent.total_episodes += 1
                Returns.append((np.array(undiscounted_rewards) * np.array(durations)).sum())
                # print(np.array(durations).sum())
                # print(undiscounted_rewards)
                # print(R)
                if param['log_level'] >= 1:
                    writer.add_scalar('Return_discrete', Returns[-1], e)
                    
                    if agent.total_episodes % 50 == 1 :
                        if 'pendulum' in param['env']:
                            agent.test_critic()
                            agent.test_value()
                        # else:
                        #     agent.plot_value()
                        #     agent.plot_policy()
                        #     agent.plot_duration()
                print(agent.total_episodes)
                if agent.total_episodes % param['save_interval'] == 0:
                    print('{} steps/s'.format(agent.total_steps//(time.time()-t0)))
                    log_data['data'] = np.array(Returns)
                    np.save('{}/{}/data/{}.npy'.format(result_path,config_name, run_ID), log_data)
                
                    agent.save_actor(filename='{}/{}/model/{}_{}.model'.format(result_path,config_name, run_ID, agent.total_episodes))
            

                break
            
            
            agent.real_t += d
            S = torch.tensor(sp, dtype=torch.float32)
            

writer.close()
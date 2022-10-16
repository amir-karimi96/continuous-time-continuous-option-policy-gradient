import torch
import time
import numpy as np
from multiprocessing import Queue
import os
import argparse
import yaml
from sub_policies import sub_policy
from env_wrapper import *# D2C, Env_test, CT_pendulum, CT_pendulum_sparse,CT_mountain_car, CT_sine, CT_sine_dense, CT_sine_vel
from torch.utils.tensorboard import SummaryWriter
from agents import *
try:
    import rlbench.gym
except:
    pass


print('imports done')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.multiprocessing.set_start_method('fork')

RB_sample_queue = Queue(maxsize=100)
agent_info_queue = Queue(maxsize=1)

Returns = []
Returns_discounted = []
Returns_times = []


parser = argparse.ArgumentParser()
parser.add_argument('--ID',default=0,type=int, help='param ID ')
parser.add_argument('--config',default='0',type=str, help='config file name')
parser.add_argument('--result_path',default='0',type=str, help='result folder path')
parser.add_argument('--load_model', default=None,type=str, help='model')

args = parser.parse_args()


run_ID = args.ID
load_model = args.load_model
cfg_path = args.config

result_path = args.result_path
with open(cfg_path) as file:
    config = yaml.full_load(file)

param = config['param']
if 'rank' in param.keys():
    rank = param['rank']
else:
    rank = 1000

torch.manual_seed(run_ID * rank) 

config_name = 'config_{}'.format(config['param_ID'])

env = globals()[param['env']](dt=np.float32(param['env_dt']))

config['state_dim'] = len(env.observation_space.sample())
action_dim = len(env.action_space.sample())
config['action_high'] = env.action_space.high[0]
config['action_low'] = env.action_space.low[0]
config['env_dt'] = np.float32(env.dt)
if param['log_level'] >= 1:
    while True:
        time.sleep(run_ID / 10)
        try:
            writer = SummaryWriter(log_dir='{}/{}/result_{}'.format(result_path,config_name, run_ID))
            break
        except:
            print("An exception occurred")
    
    if param['async'] == False:
        config['writer'] = writer
    
# if param['async']:
torch.manual_seed(run_ID * rank) 
agent = globals()[param['agent']](config, RB_sample_queue, agent_info_queue)
# else:
# agent = globals()[param['agent']](config)

low_level_function = sub_policy(low_level_function_choice = param['low_level_function'], low_level_action_dim = action_dim, n_features=int(param['z_dim']//action_dim)).low_level_function
num_ep = 500

# env.seed(rank * run_ID)
continuous_env = D2C(discrete_env= env, low_level_funciton = low_level_function, rho = agent.rho, precise=False)


# functions
def delay(duration, environment_real_time, program_real_time):
    if environment_real_time + duration > program_real_time:
        time.sleep(np.float64(environment_real_time + duration - program_real_time))

def add_data_to_RB(sample):
    if param['async']:
        RB_sample_queue.put(sample)
    else:
        agent.RB.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], z_prev = sample['z_prev'],
                        tau = sample['tau'], tau_prev = sample['tau_prev'], T = sample['T'],
                        D=sample['D'], D_prev = sample['D_prev'], d=sample['d'])

def add_data_to_RB2(sample): #TODO
    if param['async']:
        RB_sample_queue.put(sample) #TODO
    else:
        agent.RB2.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])

def add_data_to_RB3(sample):#TODO
    if param['async']:
        RB_sample_queue.put(sample)#TODO
    else:
        agent.RB3.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], z_prev = sample['z_prev'],
                        tau = sample['tau'], tau_prev = sample['tau_prev'], T = sample['T'],
                        D=sample['D'], D_prev = sample['D_prev'], d=sample['d'])

def log_data(data):

    log_data = {'config_ID': config['param_ID'], 'config': param, 'data': None, }
    
    if param['log_level'] == 2:
        writer.add_scalars('timings', {
                                        'D': data['D'],
                                        'D_sigma': data['D_sigma'],
                                        
                                        }, data['total_steps'], walltime=data['real_t'])
        # writer.add_scalars('action', {'z_mu': z,
        #                                 'z_sigma': predictions['z_sigma_target'][0],
        #                                 }, agent.total_steps, walltime=agent.real_t)
        writer.add_scalar('Reward', data['Reward'], data['total_steps'])
        writer.add_scalar('beta', data['beta'], data['total_steps'])
        if param['async']:    
            if agent_info_queue.full():
                stat_dict = agent_info_queue.get()
                
                for k, v in stat_dict.items():
                    writer.add_scalar(k, v, data['total_steps'])
        else:
            stat_dict = data['agent_data']
            if data['agent_data'] is not None: 
                for k, v in stat_dict.items():
                    writer.add_scalar(k, v, data['total_steps'])
                
    
    
    if data['done']:
        
        Returns.append((np.array(data['undiscounted_rewards']) * np.array(data['durations'])).sum())
        discounts = np.exp(-agent.rho) ** np.matmul(np.array(data['durations']), 1-np.tri(len(data['durations']), len(data['durations'])))
        #print(discounts)
        Returns_discounted.append((np.array(data['undiscounted_rewards']) * np.array(data['durations']) * discounts).sum())
        Returns_times.append(environment_real_time)
        if param['log_level'] >= 1:
            writer.add_scalar('Return_discrete', Returns[-1], data['total_episodes'])

        if data['total_episodes'] % param['save_interval'] == 0:
            # print('{} steps/s'.format(agent.total_steps//(time.time()-t0)))
            log_data['data'] = np.array(Returns)
            log_data['returns_discounted'] = np.array(Returns_discounted)
            log_data['data_wall_time'] = np.array(Returns_times)
            np.save('{}/{}/data/{}.npy'.format(result_path,config_name, run_ID), log_data)
        
            agent.save_actor(filename='{}/{}/model/{}_{}.model'.format(result_path, config_name, run_ID, data['total_episodes']))
            

if __name__ == '__main__':
    
    
    torch.manual_seed(run_ID * rank) 
    environment_real_time = 0
    program_real_time = 0
    start_update_time = None
    start_time = time.time()
    if param['async']:
        agent.update_process.start()
    max_experiment_time = param['max_experiment_time']
    agent_data = None
    
    while environment_real_time < max_experiment_time:
        
        undiscounted_rewards = []
        durations = []
        
        # reset environment
        state = continuous_env.reset()
        S = torch.tensor(state, dtype = torch.float32)
        z_prev = torch.zeros((param['z_dim'],))
        tau = 0.
        tau_prev = 0.
        D_prev = torch.tensor([0.])
        new_movement = 1
        while True:
            # get z and duration from agent
            predictions = agent.actor_network(S, z_prev, D_prev, torch.tensor([tau],dtype=torch.float32))
            
            # # sample termination
            T = torch.distributions.Categorical(torch.tensor([1-predictions['beta'], predictions['beta']])).sample()
            # T = 0
            if new_movement or T:
                if tau > 0:
                    sample2 = {'s':SS.detach().numpy(), 'sp':sp, 'r':RR, 'done_not_max':done_not_max, 'done':done,
                        'z':z.detach().numpy(), 
                        'D':D.detach().numpy(), 'd':tau}
                    add_data_to_RB2(sample2)
                    # print(sample2)
                
                if T and tau>0:
                    sample3 = sample
                    add_data_to_RB3(sample3)
            
                # sample z and duration
                # predictions = agent.actor_network(S, z_prev, D_prev, torch.tensor([tau],dtype=torch.float32))
                z, _ = agent.get_action_z(predictions['z_mu'], predictions['z_sigma'])
                D, _ = agent.get_duration(predictions['D_mu'], predictions['D_sigma'])
                
                tau = 0.
                RR = 0
                SS = S

            
            # set step duration 
            d = min((D-tau).detach().numpy()[0], agent.dt)
            # print(z,D)
            # interact with continuous environment for duration d executing z
            # sp, R, done, info = continuous_env.step(z.detach().numpy(), d, option_duration = D.detach().numpy()[0] )
            a = low_level_function(z.detach().numpy(), tau, D.detach().numpy()[0])
            sp, r, done, info = env.step(a)
            R = r * param['env_dt']
            RR += R * np.exp(-agent.rho * tau)
            
            agent.total_steps += 1
            # delay d if realtime and simulated
            program_real_time = time.time() - start_time
            if param['simulated'] and param['real_time']:
                # delay(np.array(info['durations']).sum(), environment_real_time=environment_real_time, program_real_time=program_real_time)
                delay(d, environment_real_time=environment_real_time, program_real_time=program_real_time)

            # set elapsed time it may be less tha D because of episode termination
            environment_real_time += d #np.array(info['durations']).sum()
            agent.real_t += d #np.array(info['durations']).sum()

            # apply duration penalty
            if new_movement or T:
                RR -= param['Duration_penalty_const']
            new_movement = False
            # add data to replay buffer
            done_not_max = False
            if done:
                if info['TimeLimit.truncated']==False:
                    done_not_max = True
            # print(S,R,z,done)
            sample = {'s':S.detach().numpy(), 'sp':sp, 'r':R, 'done_not_max':done_not_max, 'done':done,
                        'T':T,
                        'z':z.detach().numpy(), 'z_prev':z_prev.detach().numpy(),
                        'tau':tau, 'tau_prev':tau_prev,
                        'D':D.detach().numpy(),'D_prev':D_prev.detach().numpy(), 'd':d}
            

            add_data_to_RB(sample)

            # update info for undiscounted return computation
            undiscounted_rewards.extend([r])#info['rewards'])

            durations.extend([d])#info['durations']])
            
            # update agent if synch
            if param['async']==False:
                if agent.RB2.real_size > 1 * config['param']['batch_size'] and agent.RB3.real_size > 1 * config['param']['batch_size']:
                    if start_update_time is None:
                        start_update_time = 0. + environment_real_time
                    while agent.total_updates < config['param']['update_rate'] * (environment_real_time - start_update_time):
                        agent_data = agent.update()
                        
                        # agent.total_updates +=1
                    # print(agent.total_updates / (environment_real_time - start_update_time+ 1e-4), agent.total_updates / program_real_time)

            # log data
            data = {'agent_data': agent_data,
                'D': D.detach().numpy(),
            'D_sigma': predictions['D_sigma'].detach().numpy(),
            'total_steps': agent.total_steps,
            'total_episodes': agent.total_episodes,
            'real_t': agent.real_t,
            'Reward': R, 
            'beta': predictions['beta'].detach().numpy(),
            'undiscounted_rewards': undiscounted_rewards,
            'durations': durations,
            'done': done,
             }
            log_data(data)
            
            
            z_prev = z
            D_prev = D
            tau_prev = tau
            tau += d #np.array(info['durations']).sum()
            tau = agent.dt * round(tau/agent.dt)
            S = torch.tensor(sp, dtype=torch.float32)
            if tau + agent.dt > D:#agent.dt * round(D.detach().numpy()[0]/agent.dt):
                new_movement = True
                
            if done:
                sample2 = {'s':SS.detach().numpy(), 'sp':sp, 'r':RR, 'done_not_max':done_not_max, 'done':done,
                        'z':z.detach().numpy(), 
                        'D':D.detach().numpy(), 'd':tau}
                agent.total_episodes += 1
                break
            # print(tau, new_movement, )


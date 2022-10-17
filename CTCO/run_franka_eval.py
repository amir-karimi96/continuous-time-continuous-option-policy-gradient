from json import load
import torch
import time
import numpy as np
from multiprocessing import Queue
import torch.multiprocessing as mp
import os
import argparse
import yaml
from CTCO.utils.sub_policies import sub_policy
from CTCO.utils.env_wrapper import  D2C, Env_test, CT_pendulum, CT_pendulum_sparse,CT_mountain_car
from torch.utils.tensorboard import SummaryWriter
from CTCO.agents.agents import *
try:
    import rlbench.gym
except:
    pass
from CTCO.utils.franka_env import *

if __name__ == '__main__':
    

    print('imports done')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    # print(torch.multiprocessing.get_start_method())
    # initialize processes in 'spawn' mode, required by CUDA runtime
    ctx = mp.get_context('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RB_sample_queue = ctx.Queue(maxsize=200)
    agent_info_queue = ctx.Queue(maxsize=1)

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
    torch.manual_seed(run_ID * 1000) 
    result_path = args.result_path
    with open(cfg_path) as file:
        config = yaml.full_load(file)

    param = config['param']
    config_name = 'config_{}'.format(config['param_ID'])

    env = globals()[param['env']](dt=param['env_dt'])
    # env.seed(0)
    config['state_dim'] = len(env.observation_space.sample())
    action_dim = len(env.action_space.sample())
    config['action_high'] = env.action_space.high[0]
    config['action_low'] = env.action_space.low[0]
    config['env_dt'] = env.dt

    net_params = {  'conv': [ [3, 32, 3, 2], 
                            [32, 32, 3, 2], 
                            [32, 32, 3, 2], 
                            [32, 32, 3, 1]], 
                    'latent': 85, 
                    'mlp': [[92, 1024], [1024, 1024], [1024, 1]]}


    config['latent_dim'] = net_params['latent']
    config['image_shape'] = env.image_space.shape
    config['net_params'] = net_params
    config['device'] = device
    config['load_path'] = load_model
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
        
    if param['async']:
        agent = globals()[param['agent']](config, RB_sample_queue, agent_info_queue)
    else:
        agent = globals()[param['agent']](config)
    # print(load_model)
    # agent.load_actor(load_model)

    low_level_function = sub_policy(low_level_function_choice = param['low_level_function'], low_level_action_dim = action_dim, n_features=int(param['z_dim']//action_dim)).low_level_function
    num_ep = 500


    continuous_env = D2C(discrete_env= env, low_level_funciton = low_level_function, rho = agent.rho, precise=False)


    # functions
    def delay(duration, environment_real_time, program_real_time):
        if environment_real_time + duration > program_real_time:
            time.sleep(np.float64(environment_real_time + duration - program_real_time))

    def add_data_to_RB(sample):
        if param['async']:
            RB_sample_queue.put(sample)
        else:
            
            agent.RB.notify(s=sample['S'], image = sample['image'], sp=sample['sp'], image_p = sample['image_p'], r=sample['R'], done_not_max=sample['done_not_max'], done=sample['done'],
                            z=sample['z'], 
                            D=sample['D'], d=sample['d'])
                
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
        
            # if param['async']:    
            #     if agent_info_queue.full():
            #         stat_dict = agent_info_queue.get()
                    
            #         for k, v in stat_dict.items():
            #             writer.add_scalar(k, v, data['total_steps'])
                
        
        
        if data['done']:
        
            Returns.append((np.array(data['undiscounted_rewards']) * np.array(data['durations'])).sum())
            discounts = np.exp(-agent.rho) ** np.matmul(np.array(data['durations']), 1-np.tri(len(data['durations']), len(data['durations'])))
            #print(discounts)
            Returns_discounted.append((np.array(data['undiscounted_rewards']) * np.array(data['durations']) * discounts).sum())
            Returns_times.append(environment_real_time)
            if param['log_level'] >= 1:
                writer.add_scalar('Return_discrete', Returns[-1], data['total_episodes'])
                # if data['total_episodes'] % param['save_interval'] == 0:
                #     if agent.RB.real_size > 0:
                #         D_values = agent.RB.get_full()['D'][-500:].reshape(-1)
                #         # print(D_values)
                #         writer.add_histogram('Duration hist', D_values, global_step = agent.total_steps)
    
            if data['total_episodes'] % param['save_interval'] == 0:
                # print('{} steps/s'.format(agent.total_steps//(time.time()-t0)))
                log_data['data'] = np.array(Returns)
                log_data['returns_discounted'] = np.array(Returns_discounted)
                log_data['data_wall_time'] = np.array(Returns_times)
                np.save('{}/{}/data/{}.npy'.format(result_path,config_name, run_ID), log_data)
            
                agent.save_actor(filename='{}/{}/model/{}_{}.model'.format(result_path, config_name, run_ID, data['total_episodes']))
            
    

    
    
    environment_real_time = 0
    program_real_time = 0
    start_time = time.time()
    
    max_experiment_time = param['max_experiment_time']
    
    while environment_real_time < max_experiment_time:
        
        undiscounted_rewards = []
        durations = []
        
        # reset environment
        state = continuous_env.reset()
        # print(state)
        S_image = torch.FloatTensor(state['image']).to(device)
        S_image.unsqueeze_(0)
        S = torch.FloatTensor(state['state']).to(device)
        S.unsqueeze_(0)
        
        while True:
            # get z and duration from agent
            # print(state['image'].shape)
            
            predictions = agent.actor_network(S, S_image)
            # print(device)
            print(predictions['z_mu'], predictions['D_mu'])
            # sample z and duration
            z, _ = agent.get_action_z(predictions['z_mu'],  predictions['z_sigma'])
            D, _ = agent.get_duration(predictions['D_mu'],  predictions['D_sigma'])
            
            # set step duration to duration
            d = D.detach().cpu().numpy()[0]
            # print('d ', d)
            # print(z,D)
            # interact with continuous environment for duration d executing z
            sp, R, done, info = continuous_env.step(z.detach().cpu().numpy(), d)
            agent.total_steps += 1



            # delay d if realtime and simulated
            program_real_time = time.time() - start_time
            if param['simulated'] and param['real_time']:
                delay(np.array(info['durations']).sum(), environment_real_time=environment_real_time, program_real_time=program_real_time)

            # set elapsed time it may be less tha D because of episode termination
            environment_real_time += np.array(info['durations']).sum()
            agent.real_t += np.array(info['durations']).sum()

            # apply duration penalty
            R -= param['Duration_penalty_const']
            
            # add data to replay buffer
            done_not_max = False
            if done:
                if info['TimeLimit.truncated']==False:
                    done_not_max = True
            
            sample = {'s':S.detach().cpu().numpy(),'image': S_image.reshape(-1).detach().cpu().numpy(), 'sp':sp['state'], 'image_p': sp['image'].reshape(-1),  'r':R, 'done_not_max':done_not_max, 'done':done,
                        'z':z.detach().cpu().numpy(), 
                        'D':D.detach().cpu().numpy(), 'd':d}

            # add_data_to_RB(sample)

            # update info for undiscounted return computation
            undiscounted_rewards.extend(info['rewards'])
            durations.extend(info['durations'])
            
            # update agent if async
            # if param['async']==False:
            #     agent.update()

            # log data
            data = {'D': D.detach().cpu().numpy(),
            'D_sigma': predictions['D_sigma'].detach().cpu().numpy(),
            'total_steps': agent.total_steps,
            'total_episodes': agent.total_episodes,
            'real_t': agent.real_t,
            'Reward': R, 
            'undiscounted_rewards': undiscounted_rewards,
            'durations': durations,
            'done': done,
             }
            log_data(data)
            
            if done:
                agent.total_episodes += 1
                break
            S_image = torch.FloatTensor(sp['image']).to(device)
            S_image.unsqueeze_(0)
            S = torch.FloatTensor(sp['state']).to(device)
            S.unsqueeze_(0)


from env_wrapper import  D2C, Env_test, CT_pendulum, CT_pendulum_sparse,CT_mountain_car
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import torch
import argparse
import yaml
from PIL import Image, ImageDraw
import PIL
import os
import imageio
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from sub_policies import sub_policy
from agents import *
from multiprocessing import Queue

torch.set_printoptions(precision=10)
norm = mpl.colors.Normalize(vmin=-2, vmax=2)
cmap = cm.jet

m = cm.ScalarMappable(norm=norm, cmap=cmap)

print('imports done')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.multiprocessing.set_start_method('fork')

RB_sample_queue = Queue(maxsize=100)
agent_info_queue = Queue(maxsize=1)

Returns = []
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
env.seed(0)
config['state_dim'] = len(env.observation_space.sample())
action_dim = len(env.action_space.sample())
config['action_high'] = env.action_space.high
config['action_low'] = env.action_space.low
config['env_dt'] = env.dt
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
agent.load_actor('3_800000.model')
low_level_function = sub_policy(low_level_function_choice = param['low_level_function'], low_level_action_dim = action_dim, n_features=param['z_dim']).low_level_function
num_ep = 500
continuous_env = D2C(discrete_env= env, low_level_funciton = low_level_function, rho = agent.rho, precise=True)


def add_durations(im, x,d,color, add_line=False):
    # im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)
    # print(color)
    r,g,b,_ = color[0]
    color = (int(255*r), int(255*g), int(255*b))
    drawer.rectangle((x,100,x+d,200), fill=color )
    drawer.line((x,90,x,210), fill='gray')
    if add_line:
        drawer.line((x,80,x,220), fill='black')
    # im.show()
    return im

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def color_bar_image():
    fig = plt.figure()
    plt.colorbar(m)
    plt.savefig('color_bar.png')
    im = Image.open('color_bar.png')
    # im.show()

color_bar_image()
def delay(duration, environment_real_time, program_real_time):
    if environment_real_time + duration > program_real_time:
        time.sleep(np.float64(environment_real_time + duration - program_real_time))


def evaluate(config):
    

    environment_real_time = 0
    program_real_time = 0
    start_time = time.time()
    max_experiment_time = param['max_experiment_time']
    
    
    z_dim = param['z_dim']
    while environment_real_time < max_experiment_time:
        state = continuous_env.reset()
        S = torch.tensor(state, dtype = torch.float32)

        undiscounted_rewards = []
        durations = []
        new_movements = []
        actions = []
        frames = []
        ds = []
        colors = ['red', 'blue']
        x = 0
        while True :

            # get actor net outputs based on current state and previous option
            predictions = agent.actor_network(S)
            
            # sample z and duration
            z, _ = agent.get_action_z(predictions['z_mu'], predictions['z_sigma'])
            D, _ = agent.get_duration(predictions['D_mu'], predictions['D_sigma'])
            
            # set step duration to duration
            d = D.detach().numpy()[0]
            print(d , z)
            new_movements.append(agent.real_t)
            actions.append(0.0)
                

            # interact with continuous environment for duration d executing z
            sp, R, done, info = continuous_env.step(z.detach().numpy(), d)
            agent.total_steps += 1

            # delay d if realtime and simulated
            program_real_time = time.time() - start_time
            if param['simulated'] and param['real_time']:
                delay(d, environment_real_time=environment_real_time, program_real_time=program_real_time)
            
            # set elapsed time
            environment_real_time += d
            agent.real_t += d
            
            frames.extend(info['frames'])
            undiscounted_rewards.extend(info['rewards'])
            durations.extend(info['durations'])
            # RB_list.append([s, sp, z, R, done, D, T, tau, agent.real_t])
            # print('R: ', R)
            agent.total_steps += 1
            
            
            if done:
                agent.total_episodes += 1
                break
            S = torch.tensor(sp, dtype=torch.float32)

    return frames, durations, new_movements, actions

frames , durations, new_movements, actions = evaluate(config)
x = np.float32(0.)
n = -1
c = ['red', 'blue']
frames2 = []
im_d = PIL.Image.new(mode = "RGB", size = (1000, 200),
                           color = (255, 255, 255))
# print(len(frames), len(durations))
j = 0
# print(new_movements, durations)
for i,f in enumerate(frames):
    # print(x, new_movements[n])
    
    if np.abs(x - new_movements[n+1]) < 0.001:
        n+=1
        add_line = True
    # print(add_line, x, new_movements[n+1])
    im_d = add_durations(im_d,x*100,durations[i]*100 ,color=m.to_rgba(actions[n]), add_line=add_line)
    add_line = False
    x = x+durations[i]
    # print(x.dtype)
    frames2.append(get_concat_v(Image.fromarray(f), im_d))

imageio.mimwrite(os.path.join('./videos/', 'random_agent.mp4'), frames2, fps=30)

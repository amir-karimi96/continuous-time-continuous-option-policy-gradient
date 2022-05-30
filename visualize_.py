from env_wrapper import  CT_pendulum, D2C
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import torch
from test import COCT
import argparse
import yaml
from PIL import Image, ImageDraw
import PIL
import os
import imageio
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
torch.set_printoptions(precision=10)
norm = mpl.colors.Normalize(vmin=-2, vmax=2)
cmap = cm.jet

m = cm.ScalarMappable(norm=norm, cmap=cmap)

parser = argparse.ArgumentParser()
parser.add_argument('--ID',default=0,type=int, help='param ID ')
parser.add_argument('--config',default='0',type=str, help='config file name')
parser.add_argument('--result_path',default='0',type=str, help='result folder path')

args = parser.parse_args()
run_ID = args.ID
cfg_path = args.config
torch.manual_seed(run_ID * 1000) 
# print(cfg_path)
result_path = args.result_path
with open(cfg_path) as file:
    config = yaml.full_load(file)

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
def evaluate(config):
    env_eval = CT_pendulum()
    
    num_ep = 1
    param = config['param']
    state_dim = len(env_eval.observation_space.sample())
    action_dim = len(env_eval.action_space.sample())
    config['state_dim'] = state_dim
    agent = COCT(config)
    agent.load_actor('3_800000.model')
    continuous_env = D2C(discrete_env= env_eval, low_level_funciton= lambda x,y,z: x, rho = agent.rho)

    
    z_dim = param['z_dim']
    for e in range(num_ep):
        s = env_eval.reset()
        S = torch.tensor(s, dtype = torch.float32)
        z_prev = torch.zeros((param['z_dim'],))
        tau = 0.
        tau_prev = 0.
        D_prev = torch.tensor([0.])

        new_movement = 1
        r_newmovement = 0
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
            predictions = agent.actor_network(S,z_prev)
            
            # sample termination
            T = torch.distributions.Categorical(torch.tensor([1-predictions['beta'], predictions['beta']])).sample()
            
            if new_movement or T :
                new_movements.append(agent.real_t)
                # sample z
                z = agent.get_action_z(predictions['z_mu'], predictions['z_sigma'])
                
                # sample duration 
                D = agent.get_duration(predictions['D_mu'], predictions['D_sigma'])
                D = torch.floor(D*1000)/1000
                D = torch.round(D, decimals=3)
                tau = np.float32(0.)
                r_newmovement = -1 * 0.05
                new_movement = False
                # sample movement omega
                omega = agent.z2omega(z) # TO DO add stochasticity
                actions.append(omega.detach().numpy())
                

            d = np.float32(min((D-tau).detach().numpy()[0], agent.dt))
            # print('D: ',D)
            sp,R,done,info = continuous_env.step(omega.detach().numpy(), d)
            frames.extend(info['frames'])
            undiscounted_rewards.extend(info['rewards'])
            durations.extend(info['durations'])
            # RB_list.append([s, sp, z, R, done, D, T, tau, agent.real_t])
            # print('R: ', R)
            agent.total_steps += 1
            
            
            if done:
                break
            
            z_prev = z
            D_prev = D
            tau_prev = tau
            tau += d
            agent.real_t += d
            # print(info['durations'], d)
            S = torch.tensor(sp, dtype=torch.float32)
            # print(tau, D)
            if tau >= D:
                new_movement = True
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

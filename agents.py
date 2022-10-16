import torch 
import numpy as np
from env_wrapper import  D2C, Env_test, CT_pendulum
import gym
import argparse
import yaml
import matplotlib.pyplot as plt
from dataset import Variable, Domain, RLDataset
from torch.utils.tensorboard import SummaryWriter
import time

from matplotlib import cm, image
import pickle
import multiprocessing
from torch.nn import Parameter
import torch.nn as nn
# import cv2
import torch.multiprocessing as mp

def neural_net(input_size, output_size, num_hidden_layer, hidden_layer_size, activation):
    NN = []
    if num_hidden_layer > 0:
        NN.append(torch.nn.Linear(input_size, hidden_layer_size))
        NN.append(getattr(torch.nn, activation)())
        for i in range(num_hidden_layer-1):
            NN.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            NN.append(getattr(torch.nn, activation)())
        NN.append(torch.nn.Linear(hidden_layer_size, output_size))
    else:
        NN.append(torch.nn.Linear(input_size, output_size))
    NN = torch.nn.Sequential(*NN)
    return NN

class COCT_actor_network(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        param = config['param']
        self.config = config
        self.std_depend_on_state = param['std_depend_on_state']
        
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        self.actor_z_mu = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_D_sigma = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_z_mu_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu_target = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma_target.add_module('softplus',torch.nn.Softplus())

        self.actor_D_sigma_target = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma_target.add_module('softplus',torch.nn.Softplus())
        
        self.actor_z_mu_target.load_state_dict(self.actor_z_mu.state_dict())
        self.actor_D_mu_target.load_state_dict(self.actor_D_mu.state_dict())
        self.actor_z_sigma_target.load_state_dict(self.actor_z_sigma.state_dict())
        self.actor_D_sigma_target.load_state_dict(self.actor_D_sigma.state_dict())
        
        # option terminaiton model beta(T|s,z_{-1})
        self.beta = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] + param['z_dim'] + 1 + 1, param['beta_NN_nhid']), getattr(torch.nn, param['beta_NN_gate'])(),
                                torch.nn.Linear(param['beta_NN_nhid'], param['beta_NN_nhid']), getattr(torch.nn, param['beta_NN_gate'])(),
                                torch.nn.Linear(param['beta_NN_nhid'], 1),torch.nn.Sigmoid())
        
        # parameter lists
        self.actor_z_params = list(self.actor_z_mu.parameters())+ list(self.actor_z_sigma.parameters())
        self.actor_D_params = list(self.actor_D_mu.parameters())+ list(self.actor_D_sigma.parameters())
        self.actor_beta_params = list(self.beta.parameters())

        self.actor_optimizer = torch.optim.Adam([
                {'params': self.actor_z_params, 'lr': param['actor_z_lr']},
                {'params': self.actor_D_params, 'lr': param['actor_D_lr']},
                {'params': self.actor_beta_params, 'lr': param['actor_beta_lr']}
            ])

        # freez nets if needed
        if param['freez_z']:
            self.actor_z_mu.requires_grad_(False)
            self.actor_z_sigma.requires_grad_(False)

        if param['freez_D']:
            self.actor_D_mu.requires_grad_(False)
            self.actor_D_sigma.requires_grad_(False)

        if param['freez_beta']:
            self.beta.requires_grad_(False)

    def forward(self,state, z_,D_, tau): 
        # inputs:   current state
        #           previous option z_
        z_mu= self.actor_z_mu(state)
        D_mu= self.actor_D_mu(state)
        
        z_sigma = self.actor_z_sigma(state)
        D_sigma = self.actor_D_sigma(state)
        
        z_mu_target= self.actor_z_mu_target(state)
        D_mu_target= self.actor_D_mu_target(state)
        
        z_sigma_target = self.actor_z_sigma_target(state)
        D_sigma_target = self.actor_D_sigma_target(state)
        # print(1111)
        # print(state, z_, tau)
        beta = 1.0 / ( 1.0 + self.beta(torch.cat((state, z_,D_, tau), dim=-1)) / self.config['env_dt'])
        # print(self.beta(torch.cat((state, z_, tau), dim=-1)))
        predictions = { 'beta': beta,
                        'z_mu':  z_mu ,
                        'z_sigma': z_sigma ,
                        'D_mu':   D_mu ,
                        'D_sigma':  D_sigma ,
                        'z_mu_target':  z_mu_target,
                        'z_sigma_target': z_sigma_target ,
                        'D_mu_target':  D_mu_target,
                        'D_sigma_target': D_sigma_target ,
                        # 'omega': self.z2omega( z_D[...,:-1] ),
                        }
        return predictions


class COCT_actor_network_simple(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        param = config['param']
        self.std_depend_on_state = param['std_depend_on_state']
        
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        self.actor_z_mu = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_D_sigma = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_z_mu_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu_target = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma_target.add_module('softplus',torch.nn.Softplus())

        self.actor_D_sigma_target = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma_target.add_module('softplus',torch.nn.Softplus())
        
        self.actor_z_mu_target.load_state_dict(self.actor_z_mu.state_dict())
        self.actor_D_mu_target.load_state_dict(self.actor_D_mu.state_dict())
        self.actor_z_sigma_target.load_state_dict(self.actor_z_sigma.state_dict())
        self.actor_D_sigma_target.load_state_dict(self.actor_D_sigma.state_dict())
        
        

        # parameter lists
        self.actor_z_params = list(self.actor_z_mu.parameters())+ list(self.actor_z_sigma.parameters())
        self.actor_D_params = list(self.actor_D_mu.parameters())+ list(self.actor_D_sigma.parameters())

        self.actor_optimizer = torch.optim.Adam([
                {'params': self.actor_z_params, 'lr': param['actor_z_lr']},
                {'params': self.actor_D_params, 'lr': param['actor_D_lr']},
                
            ])

        # freez nets if needed
        if param['freez_z']:
            self.actor_z_mu.requires_grad_(False)
            self.actor_z_sigma.requires_grad_(False)

        if param['freez_D']:
            self.actor_D_mu.requires_grad_(False)
            self.actor_D_sigma.requires_grad_(False)


    def forward(self,state):
        # inputs:   current state
        #           previous option z_
        z_mu= self.actor_z_mu(state)
        D_mu= self.actor_D_mu(state)
        
        z_sigma = self.actor_z_sigma(state)
        D_sigma = self.actor_D_sigma(state)
        
        z_mu_target= self.actor_z_mu_target(state)
        D_mu_target= self.actor_D_mu_target(state)
        
        z_sigma_target = self.actor_z_sigma_target(state)
        D_sigma_target = self.actor_D_sigma_target(state)
        
        predictions = { 
                        'z_mu': z_mu ,
                        'z_sigma': z_sigma ,
                        'D_mu':   D_mu ,
                        'D_sigma':  D_sigma ,
                        'z_mu_target': z_mu_target,
                        'z_sigma_target': z_sigma_target ,
                        'D_mu_target':  D_mu_target ,
                        'D_sigma_target': D_sigma_target ,
                        # 'omega': self.z2omega( z_D[...,:-1] ),
                        }
        return predictions


class FiGAR_actor_network(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        param = config['param']
        self.std_depend_on_state = param['std_depend_on_state']
        
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        self.actor_z_mu = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_prefs = neural_net(input_size= config['state_dim'], output_size = param['D_max'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_z_mu_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_prefs_target = neural_net(input_size= config['state_dim'], output_size= param['D_max'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma_target.add_module('softplus',torch.nn.Softplus())

        
        self.actor_z_mu_target.load_state_dict(self.actor_z_mu.state_dict())
        self.actor_D_prefs_target.load_state_dict(self.actor_D_prefs.state_dict())
        self.actor_z_sigma_target.load_state_dict(self.actor_z_sigma.state_dict())
        
        

        # parameter lists
        self.actor_z_params = list(self.actor_z_mu.parameters())+ list(self.actor_z_sigma.parameters())
        self.actor_D_params = list(self.actor_D_prefs.parameters())

        self.actor_optimizer = torch.optim.Adam([
                {'params': self.actor_z_params, 'lr': param['actor_z_lr']},
                {'params': self.actor_D_params, 'lr': param['actor_D_lr']},
                
            ])

        # freez nets if needed
        if param['freez_z']:
            self.actor_z_mu.requires_grad_(False)
            self.actor_z_sigma.requires_grad_(False)

        if param['freez_D']:
            self.actor_D_prefs.requires_grad_(False)


    def forward(self,state):
        # inputs:   current state
        #           previous option z_
        z_mu= self.actor_z_mu(state)
        D_prefs= self.actor_D_prefs(state)
        
        z_sigma = self.actor_z_sigma(state)
        
        z_mu_target= self.actor_z_mu_target(state)
        D_prefs_target= self.actor_D_prefs_target(state)
        
        z_sigma_target = self.actor_z_sigma_target(state)
        
        
        predictions = { 
                        'z_mu': z_mu ,
                        'z_sigma': z_sigma ,
                        'D_prefs':   D_prefs ,
                        'z_mu_target': z_mu_target,
                        'z_sigma_target': z_sigma_target ,
                        'D_prefs_target': D_prefs_target ,
                        # 'omega': self.z2omega( z_D[...,:-1] ),
                        }
        return predictions


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def conv_out_size(input_size, kernel_size, stride, padding=0):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.contiguous().view(-1, self.height*self.width)

        softmax_attention = torch.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints


class EncoderModel(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax=True):
        super().__init__()

        if image_shape[-1] != 0: # use image
            h, w, c = image_shape
            # print(h, w, c)

            self.rad_h = round(rad_offset * h)
            self.rad_w = round(rad_offset * w)
            image_shape = (c, h-2*self.rad_h, w-2*self.rad_w)
            self.init_conv(image_shape, net_params)
            
            if spatial_softmax:
                self.latent_dim = net_params['conv'][-1][1] * 2
            else:
                self.latent_dim = net_params['latent']
            
            if proprioception_shape[-1] == 0: # no proprioception readings
                self.encoder_type = 'pixel'
                
            else: # image with proprioception
                self.encoder_type = 'multi' 
                self.latent_dim += proprioception_shape[0]

        elif proprioception_shape[-1] != 0:
            self.encoder_type = 'proprioception'
            self.latent_dim = proprioception_shape[0]

        else:
            raise NotImplementedError('Invalid observation combination')
        

    def init_conv(self, image_shape, net_params):
        conv_params = net_params['conv']
        latent_dim = net_params['latent']
        channel, height, width = image_shape
        conv_params[0][0] = channel
        layers = []
        for i, (in_channel, out_channel, kernel_size, stride) in enumerate(conv_params):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride))
            if i < len(conv_params) - 1:
                layers.append(nn.ReLU())
            width = conv_out_size(width, kernel_size, stride)
            height = conv_out_size(height, kernel_size, stride)

        self.convs = nn.Sequential(
            *layers
        )
        self.ss = SpatialSoftmax(width, height, conv_params[-1][1])
        self.fc = nn.Linear(conv_params[-1][1] * width * height, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.apply(weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        if self.encoder_type == 'proprioception':
            return proprioceptions

        if self.encoder_type == 'pixel' or self.encoder_type == 'multi':
            images = images / 255.
            if random_rad:
                images = random_augment(images, self.rad_h, self.rad_w)
            else:
                if (len(images.shape) == 3):
                    n, h, w = images.shape
                    c = 1
                else:
                    n, h, w, c = images.shape
                    # print(n, h, w, c)
                    images = images[:,
                    self.rad_h : h-self.rad_h,
                    self.rad_w : w-self.rad_w, 
                    :
                    ]

                images = torch.reshape(images, (n, c, h-2*self.rad_h ,w-2*self.rad_w))

                # print("image size", images.shape)

            h = self.ss(self.convs(images))
            if detach:
                h = h.detach()

            if self.encoder_type == 'multi':

                h = torch.cat([h, proprioceptions], dim=-1)

            return h
        else:
            raise NotImplementedError('Invalid encoder type')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(0.1)
        m.bias.data.fill_(0.01)

class visual_CTCO_network(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = EncoderModel(config['image_shape'], [config['state_dim']], config['net_params'], rad_offset=0)
        self.encoder.to(config['device'])
        config['latent_dim'] = self.encoder.latent_dim
        param = config['param']
        self.std_depend_on_state = param['std_depend_on_state']
        
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        self.actor_z_mu = neural_net(input_size= config['latent_dim'] , output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_mu.apply(init_weights)
        self.actor_D_mu = neural_net(input_size= config['latent_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma = neural_net(input_size= config['latent_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma.apply(init_weights)
        self.actor_z_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_D_sigma = neural_net(input_size= config['latent_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma.apply(init_weights)

        self.actor_D_sigma.add_module('softplus',torch.nn.Softplus())

        self.actor_z_mu_target = neural_net(input_size= config['latent_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu_target = neural_net(input_size= config['latent_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma_target = neural_net(input_size= config['latent_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma_target.add_module('softplus',torch.nn.Softplus())

        self.actor_D_sigma_target = neural_net(input_size= config['latent_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma_target.add_module('softplus',torch.nn.Softplus())
        
        self.actor_z_mu_target.load_state_dict(self.actor_z_mu.state_dict())
        self.actor_D_mu_target.load_state_dict(self.actor_D_mu.state_dict())
        self.actor_z_sigma_target.load_state_dict(self.actor_z_sigma.state_dict())
        self.actor_D_sigma_target.load_state_dict(self.actor_D_sigma.state_dict())
        
        

        # parameter lists
        self.actor_z_params = list(self.actor_z_mu.parameters())+ list(self.actor_z_sigma.parameters()) + list(self.encoder.parameters())
        self.actor_D_params = list(self.actor_D_mu.parameters())+ list(self.actor_D_sigma.parameters())

        self.actor_optimizer = torch.optim.Adam([
                {'params': self.actor_z_params, 'lr': param['actor_z_lr']},
                {'params': self.actor_D_params, 'lr': param['actor_D_lr']},
                
            ])

        # freez nets if needed
        if param['freez_z']:
            self.actor_z_mu.requires_grad_(False)
            self.actor_z_sigma.requires_grad_(False)

        if param['freez_D']:
            self.actor_D_mu.requires_grad_(False)
            self.actor_D_sigma.requires_grad_(False)

        print(config['state_dim'])
        

    def forward(self,proprioceptions, images):
        # inputs:   current state
        #           previous option z_
        random_rad = False
        detach_encoder = False
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        # print(latents.shape)

        z_mu= self.actor_z_mu(latents)
        D_mu= self.actor_D_mu(latents)
        
        z_sigma = self.actor_z_sigma(latents)
        D_sigma = self.actor_D_sigma(latents)
        
        z_mu_target= self.actor_z_mu_target(latents)
        D_mu_target= self.actor_D_mu_target(latents)
        
        z_sigma_target = self.actor_z_sigma_target(latents)
        D_sigma_target = self.actor_D_sigma_target(latents)

        predictions = { 
                        'z_mu': z_mu ,
                        'z_sigma': z_sigma ,
                        'D_mu':   D_mu ,
                        'D_sigma':  D_sigma ,
                        'z_mu_target': z_mu_target,
                        'z_sigma_target': z_sigma_target ,
                        'D_mu_target': D_mu_target ,
                        'D_sigma_target': D_sigma_target ,
                        # 'omega': self.z2omega( z_D[...,:-1] ),
                        }
        return predictions




class COCT:
    def __init__(self,config, RB_sample_queue, agent_info_queue) -> None:
        self.writer = config['writer']
        self.config = config
        
        self.rho = 0.4#- np.log(config['param']['discount']) / self.config['env_dt']
        self.dt = self.config['env_dt'] # TODO may change
        param = config['param']
        
        self.actor_network = COCT_actor_network(config)
        self.actor_network.share_memory()

        # critic model inputs are state, z, tau, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1 + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1 + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])


        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        

        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        
        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=param['log_alpha_lr'],)
        # z dim + duration dim
        self.target_entropy = -(param['z_dim']+1)
        
        
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']), Variable('z_prev',param['z_dim']),
                Variable('T', 1),
                Variable('tau',1), Variable('tau_prev',1),
                Variable('D',1), Variable('D_prev',1),
                Variable('d',1), Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
        D2 = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))  

        D3 = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']), Variable('z_prev',param['z_dim']),
                Variable('T', 1),
                Variable('tau',1), Variable('tau_prev',1),
                Variable('D',1), Variable('D_prev',1),
                Variable('d',1), Variable('r',1), Variable('done',1), Variable('done_not_max',1))  

        self.RB = RLDataset(D)
        self.RB2 = RLDataset(D2)
        self.RB3 = RLDataset(D3)

 
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        
                
        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])


        # self.z2omega = lambda x: x
        #self.update_process = multiprocessing.Process(target=self.update_process_target, args=(RB_sample_queue,A_Q,config), daemon=True)

    def update_process_target(self,R_sample_Q,A_Q,config):
        counter = 0

        while True:
            counter +=1
            # print('counter',counter)
            while not R_sample_Q.empty():
                sample = R_sample_Q.get()
                # print(sample)
                #TODO add other variables
                self.RB.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])
                # print('RB: ', self.RB.real_size)
            # tt = time.time()
            # time.sleep(0.1)

            self.update_async(A_Q,counter) 
    
    def update_async(self,A_Q,counter):
        if self.RB.real_size > 1 * self.config['param']['batch_size']:
            # print('update') 
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
            self.total_updates += 1
            param = self.config['param']
            # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
            agent_data = {}
            agent_data['critic_loss'] = c_loss
            agent_data['alpha'] = self.alpha.detach().numpy()
            if A_Q.empty():
                A_Q.put(agent_data)
        else:
            
            time.sleep(0.01)


    def get_action_z(self, mu_z, sigma_z):
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z

    def get_duration(self, mu_D, sigma_D):
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        D0 = D_dist.rsample()
        # D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        # D = D0 ** 2 + self.config['env_dt']
        D = torch.sigmoid(D0) + self.config['env_dt']
        # D = D*0 + 1.0
        # log_prob_D = (D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        # if square of gaussian
        #fD_D:
        # log_prob_D = torch.log(1/(2 * (1.0e-6 + D-self.config['env_dt'])**0.5) * (torch.exp(D_dist.log_prob(D0)) + torch.exp(D_dist.log_prob(-D0))))
        
        # if sigmoid
        # FD_D:
        log_prob_D = (D_dist.log_prob(D0) - torch.log(1e-6 + torch.sigmoid(D0) * (1 - torch.sigmoid(D0)) )).sum(-1)
        return D, log_prob_D


    def update_critic(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)

        Taus = torch.tensor(data['tau'], dtype=torch.float32)
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)
        dones_not_max = torch.tensor(data['done_not_max'], dtype=torch.float32)

        predictions = self.actor_network(SPs,Zs, Ds , Taus+self.dt) # TODO check for self.dt == d
        
        # sample terminations
        TPs = torch.distributions.Categorical(torch.cat([1-predictions['beta'], predictions['beta']], dim=-1)).sample().reshape(-1,1)
        # determine newmovement
        new_movementPs = ((Taus + 2*self.dt) > Ds)

        new_movementPs = new_movementPs + (TPs==1) # newmovement or T
        new_movementPs = new_movementPs * 1.

        ZPs, log_probs_ZPs = self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4)
        ZPs = Zs * (new_movementPs==0) + (new_movementPs==1) * ZPs

        DPs, log_probs_DPs = self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)
        DPs = Ds * (new_movementPs==0) + (new_movementPs==1) * DPs

        TauPs = (Taus + ds) * (new_movementPs==0) + 0 * (new_movementPs==1)
        
        SPs_ZPs_DPs_TauPs = torch.cat((SPs,ZPs,DPs,TauPs), dim=1)
        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs,Ds,Taus), dim=1)
        
        # print(torch.cat((Ss,Rs),dim = -1))
        # print(Ss_Zs_Ds_Taus)
        log_probs = log_probs_ZPs + log_probs_DPs # TODO see what does it mean, when there is no new movement
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs_TauPs) - new_movementPs * self.alpha.detach() * log_probs # TODO like sac add another critic network
        

        target_Qs = Rs + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V 
        current_Qs = self.critic(Ss_Zs_Ds_Taus.detach())
        # print(torch.cat((target_Qs, Rs),dim=-1))
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        # value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs_TauPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_critic2(self, data2, data3):
        np.concatenate((data2['s'], data3['s']), axis=0)
        Ss = torch.tensor(np.concatenate((data2['s'], data3['s']), axis=0), dtype=torch.float32)
        SPs = torch.tensor(np.concatenate((data2['sp'], data3['sp']), axis=0), dtype=torch.float32)
        Zs = torch.tensor(np.concatenate((data2['z'], data3['z']), axis=0), dtype=torch.float32)
        Ds = torch.tensor(np.concatenate((data2['D'], data3['D']), axis=0), dtype=torch.float32)
        ds = torch.tensor(np.concatenate((data2['d'], data3['d']), axis=0), dtype=torch.float32)

        Taus = torch.tensor(np.concatenate((0 * data2['D'], data3['tau']), axis=0), dtype=torch.float32)
        Rs = torch.tensor(np.concatenate((data2['r'], data3['r']), axis=0), dtype=torch.float32)
        dones = torch.tensor(np.concatenate((data2['done'], data3['done']), axis=0), dtype=torch.float32)
        dones_not_max = torch.tensor(np.concatenate((data2['done_not_max'], data3['done_not_max']), axis=0), dtype=torch.float32)
        predictions = self.actor_network(SPs,Zs, Ds , 0*Ds) # TODO check for self.dt == d
        
        
        ZPs, log_probs_ZPs = self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4)
        log_probs_ZPs = log_probs_ZPs.reshape(-1,1)

        DPs, log_probs_DPs = self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)
        log_probs_DPs = log_probs_DPs.reshape(-1,1)

        TauPs =  0 * Taus
        
        SPs_ZPs_DPs_TauPs = torch.cat((SPs,ZPs,DPs,TauPs), dim=1)
        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs,Ds,Taus), dim=1)
        
        # print(torch.cat((Ss,Rs),dim = -1))
        # print(Ss_Zs_Ds_Taus)
        log_probs = log_probs_ZPs + log_probs_DPs # TODO see what does it mean, when there is no new movement
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs_TauPs) -  self.alpha.detach() * log_probs # TODO like sac add another critic network
        

        target_Qs = Rs + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V 

        current_Qs = self.critic(Ss_Zs_Ds_Taus.detach())
        # print(torch.cat((target_Qs, Rs),dim=-1))
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        # value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs_TauPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Zs_prev = torch.tensor(data['z_prev'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ds_prev = torch.tensor(data['D_prev'], dtype=torch.float32)
        Taus = torch.tensor(data['tau'], dtype=torch.float32)
        Taus_prev = torch.tensor(data['tau_prev'], dtype=torch.float32)

        predictions = self.actor_network(Ss,Zs_prev,Ds_prev, Taus) #TODO check

        q = 1*(Taus_prev + 2 * self.dt < Ds_prev)  # important !!! since the implementation is not precise in time we need to add 2*dt not dt 
        # print(q)
        k =  (1-q) + q * predictions['beta'].detach()
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        

        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs, Ds, 0*Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds_Taus)
        # print(k * Ss_Zs_Ds_Taus)

        z_loss = k * (self.alpha.detach() * log_prob_Zs - current_Qs_z) # TODO check
               
        # duration loss
        Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        Ds = Ds.reshape(-1,1)

        Zs = torch.tensor(data['z'], dtype=torch.float32)
        # Ss_Zs_Taus_Ds = torch.cat((Ss,Zs,(0*Ds).detach(), Ds), dim=1)
        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs, Ds,(0*Ds).detach()), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds_Taus)

        D_loss =  k * (self.alpha.detach() * log_prob_Ds - current_Qs_D) # TODO check

        # beta loss
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs, Ds, 0*Ds), dim=1)
        Qs = self.critic(Ss_Zs_Ds_Taus)
        Ss_Zs__Ds__Taus = torch.cat((Ss, Zs_prev, Ds_prev, Taus_prev + self.dt), dim=1)
        Qs_ = self.critic(Ss_Zs__Ds__Taus)
        
        beta_loss = - q * ( predictions['beta'] * (Qs - Qs_).detach() - self.alpha.detach() * (log_prob_Zs + log_prob_Ds).detach())
        # print(torch.cat((q,Taus)))
        # update all
        # print(z_loss)
        loss = (z_loss + D_loss + beta_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        # self.beta_opt.zero_grad()
        loss.backward()
        self.actor_network.actor_optimizer.step()
        # self.beta_opt.step()
        # if param['log_level'] == 2:
        #     writer.add_scalars('z_D_batch_size', {'critic':(1-q).sum().detach(),
        #                                         }, self.total_steps, walltime=self.real_t)
        
        # TODO check
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Ds+log_prob_Zs) - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() , beta_loss.mean().detach().numpy() )
    
    def update_actors2_z_D_alpha(self, data):
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Zs_prev = torch.zeros((self.config['param']['batch_size'], self.config['param']['z_dim']))
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ds_prev = torch.zeros((self.config['param']['batch_size'], 1))
        Taus = torch.zeros((self.config['param']['batch_size'], 1))
        self.config
        predictions = self.actor_network(Ss,Zs_prev,Ds_prev, Taus) #TODO check

        #q = 1*(Taus_prev + 2 * self.dt < Ds_prev)  # important !!! since the implementation is not precise in time we need to add 2*dt not dt 
        # print(q)
        #k =  (1-q) + q * predictions['beta'].detach()
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)

        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs, Ds, 0*Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds_Taus)
        # print(k * Ss_Zs_Ds_Taus)

        z_loss = (self.alpha.detach() * log_prob_Zs - current_Qs_z) # TODO check


        # duration loss
        Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        Ds = Ds.reshape(-1,1)
        log_prob_Ds = log_prob_Ds.reshape(-1,1)

        Zs = torch.tensor(data['z'], dtype=torch.float32)
        # Ss_Zs_Taus_Ds = torch.cat((Ss,Zs,(0*Ds).detach(), Ds), dim=1)
        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs, Ds,(0*Ds).detach()), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds_Taus)

        D_loss =  (self.alpha.detach() * log_prob_Ds - current_Qs_D) # TODO check


        # TODO check
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Ds+log_prob_Zs) - self.target_entropy).detach()).mean()
        
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        return z_loss, D_loss

    def update_actors2_beta(self, data):
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Zs_prev = torch.tensor(data['z_prev'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ds_prev = torch.tensor(data['D_prev'], dtype=torch.float32)
        Taus = torch.tensor(data['tau'], dtype=torch.float32)
        Taus_prev = torch.tensor(data['tau_prev'], dtype=torch.float32)

        predictions = self.actor_network(Ss,Zs_prev,Ds_prev, Taus) #TODO check
        
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)
        Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        Ds = Ds.reshape(-1,1)
        log_prob_Ds = log_prob_Ds.reshape(-1,1)
        # beta loss
        # Zs = torch.tensor(data['z'], dtype=torch.float32)
        # Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ss_Zs_Ds_Taus = torch.cat((Ss,Zs, Ds, 0*Ds), dim=1)
        Qs = self.critic(Ss_Zs_Ds_Taus)
        Ss_Zs__Ds__Taus = torch.cat((Ss, Zs_prev, Ds_prev, Taus_prev + self.dt), dim=1)
        Qs_ = self.critic(Ss_Zs__Ds__Taus)
        
        beta_loss = - ( torch.log(predictions['beta']) * (Qs - Qs_).detach() - self.alpha.detach() * (log_prob_Zs + log_prob_Ds).detach())
        return beta_loss
    
    def update_actors2(self, data2, data3):
        z_loss, D_loss = self.update_actors2_z_D_alpha(data2)
        beta_loss = self.update_actors2_beta(data3)
        loss = (z_loss + D_loss + beta_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        # self.beta_opt.zero_grad()
        loss.backward()
        self.actor_network.actor_optimizer.step()
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() , beta_loss.mean().detach().numpy() )

    def update(self):
        data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        data2 = self.RB2.get_minibatch(size = self.config['param']['batch_size'])
        data3 = self.RB3.get_minibatch(size = self.config['param']['batch_size'])
        self.total_updates += 1
        param = self.config['param']
        
        #####
        c_loss = self.update_critic2(data2, data3)
        #####

        self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
        
        # z_loss, D_loss, beta_loss = self.update_actors(data)   
        #####
        z_loss, D_loss, beta_loss = self.update_actors2(data2, data3)  
        #####
        self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        agent_data = {}
        agent_data['critic_loss'] = c_loss
        agent_data['alpha'] = self.alpha.detach().numpy()
        # if self.total_updates % 1000 == 0:
        #     self.test_critic()
        return agent_data

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-1, 1, 100)
        D = torch.linspace(0, 1, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        T = 0. * D_
        S = 0. * Z_
        print(S.shape,Z_.shape,T.shape,D_.shape)
        SZDT = torch.cat((S,Z_,D_,T),dim=-1)
        Y = self.critic(SZDT).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States, States*0)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States, States*0)
        Y = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()

class COCT_simple:
    def __init__(self,config) -> None:
        self.writer = config['writer']
        self.config = config
        
        self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        param = config['param']
        # critic model inputs are state, z, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic2 = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])


        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        
        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic2[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic2[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic2[-5].weight, nonlinearity='relu') 
            self.critic2[-1].bias.data[:] = 0*torch.rand(self.critic2[-1].bias.data[:].shape)-0
            self.critic2[-3].bias.data[:] = 2*torch.rand(self.critic2[-3].bias.data[:].shape)-1
            self.critic2[-5].bias.data[:] = 2*torch.rand(self.critic2[-5].bias.data[:].shape)-1

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic2[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic2[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic2[-5].weight, nonlinearity='relu') 
            self.critic2[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic2[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic2[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_network = COCT_actor_network_simple(config)
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        
        
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1))    
    

        self.RB = RLDataset(D)
        self.total_steps = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])


        self.z2omega = lambda x: x

    def get_action_z(self, mu_z, sigma_z):
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        
        return z
        
    def get_duration(self, mu_D, sigma_D):
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        D0 = D_dist.rsample()
        D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        
        return D

    def update_critic(self, data):    

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)

        
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)

        predictions = self.actor_network(SPs)
        
        
        ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 

        DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4).reshape(-1,1)

        SPs_ZPs_DPs = torch.cat((SPs,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        
        target_Qs = Rs + torch.exp(-self.rho * ds) *  self.critic_target(SPs_ZPs_DPs) # TODO add different dones ...
        current_Qs = self.critic(Ss_Zs_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        
        predictions = self.actor_network(Ss)

        
        
        # high-policy loss
        Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
       

        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = - current_Qs_z
               
        # duration loss
        Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma']).reshape(-1,1)
        
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds)
        D_loss = - current_Qs_D


        # update all
        # print(z_loss)
        loss = (z_loss + D_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()
        
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() )

    def update(self, num_epochs=1):
        param = self.config['param']
        
        for i in range(num_epochs):
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        if self.total_steps % 10000 == 5000:
            D_values = self.RB.get_full()['D'][-5000:].reshape(-1)
            # print(D_values)
            if param['log_level']>=1:
                self.writer.add_histogram('Duration hist', D_values, global_step = self.total_steps)
        
        
        # z_loss = D_loss = beta_loss = 0
        if param['log_level']==2:
            self.writer.add_scalars('losses', {'critic':c_loss,
                                            'actor_z': z_loss,
                                            'actor_D': D_loss,
                                            }, self.total_steps, walltime=self.real_t)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        # print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

class COCT_SAC:
    def __init__(self,config) -> None:
        self.writer = config['writer']
        self.config = config
        
        self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        param = config['param']
        # critic model inputs are state, z, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target.load_state_dict(self.critic.state_dict())

        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.actor_network = COCT_actor_network_simple(config)
        
        self.log_alpha = torch.tensor(np.log(self.config['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha.requires_grad = self.config['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.config['log_alpha_lr'],)
        # z dim + duration dim
        self.target_entropy = -(param['z_dim']+1)
        
        
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    

        self.RB = RLDataset(D)
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])


        self.z2omega = lambda x: x

    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        D0 = D_dist.rsample()
        D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        log_prob_D = (D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        
        return D, log_prob_D

    def update_critic(self, data):

        
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)

        
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)

        predictions = self.actor_network(SPs)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 

        DPs, log_probs_DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)

        SPs_ZPs_DPs = torch.cat((SPs,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        
        log_probs = log_probs_ZPs + log_probs_DPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs # TODO like sac add another critic network
        

        target_Qs = Rs + torch.exp(-self.rho * ds) *  target_V # TODO add different dones ...
        current_Qs = self.critic(Ss_Zs_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        
        predictions = self.actor_network(Ss)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
       

        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        # duration loss
        Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        Ds = Ds.reshape(-1,1)
        
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds)
        
        D_loss = self.alpha.detach() * log_prob_Ds - current_Qs_D

        

        # update all
        # print(z_loss)
        loss = (z_loss + D_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()

        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Ds+log_prob_Zs) - self.target_entropy).detach()).mean()
            self.writer.add_scalar('alpha',self.alpha, global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() )

    def update(self, num_epochs=1):
        param = self.config['param']
        
        for i in range(num_epochs):
            self.total_updates += 1
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        if self.total_steps % 2000 == 500 and param['log_level']>=1:
            D_values = self.RB.get_full()['D'][-500:].reshape(-1)
            # print(D_values)
            self.writer.add_histogram('Duration hist', D_values, global_step = self.total_steps)
        
        
        # z_loss = D_loss = beta_loss = 0
        if param['log_level']==2:
            self.writer.add_scalars('losses', {'critic':c_loss,
                                            'actor_z': z_loss,
                                            'actor_D': D_loss,
                                            }, self.total_updates, walltime=self.real_t)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()

class COCT_SAC_async:
    def __init__(self,config, RB_sample_queue, A_Q) -> None:
        self.writer = config['writer']
        self.config = config
        self.save_time = 0
        # self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        self.rho = 0.4 # its good for 10 sec episodes TODO better explain why.
        param = config['param']
        if 'duration_max' in param:
            self.duration_max = param['duration_max']
        else:
            self.duration_max = 1.0


        self.actor_network = COCT_actor_network_simple(config)
        self.actor_network.share_memory()
        # critic model inputs are state, z, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        
        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        

        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_D = torch.tensor(np.log(param['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha_D.requires_grad = param['log_alpha_requires_grad']
        
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha, self.log_alpha_D],
                                                    lr=param['log_alpha_lr'],)
        # z dim + duration dim
        self.target_entropy = -param['z_dim']
        self.target_entropy_D = -1
        
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    

        self.RB = RLDataset(D)

        # self.z2omega = lambda x: x
        self.update_process = multiprocessing.Process(target=self.update_process_target, args=(RB_sample_queue,A_Q,config), daemon=True)


    def update_process_target(self,R_sample_Q,A_Q,config):
        counter = 0

        while True:
            counter +=1
            # print('counter',counter)
            while not R_sample_Q.empty():
                sample = R_sample_Q.get()
                # print(sample)
                self.RB.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])
                # print('RB: ', self.RB.real_size)
            # tt = time.time()
            # time.sleep(0.1)

            self.update_async(A_Q,counter) 
    
    def update_async(self,A_Q,counter):
        if self.RB.real_size > 1 * self.config['param']['batch_size']:
            # print('update') 
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
            self.total_updates += 1
            param = self.config['param']
            # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
            agent_data = {}
            agent_data['critic_loss'] = c_loss
            agent_data['alpha'] = self.alpha.detach().numpy()
            if A_Q.empty():
                A_Q.put(agent_data)
        else:
            
            time.sleep(0.01)


    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        # shift = -1
        D0 = D_dist.rsample()#+shift
        # D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        # D = D0 ** 2 + self.config['env_dt']
        D = self.duration_max * torch.sigmoid(D0) + self.config['env_dt']
        # D = D*0 + 1.0
        # log_prob_D = (D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        # if square of gaussian
        #fD_D:
        # log_prob_D = torch.log(1/(2 * (1.0e-6 + D-self.config['env_dt'])**0.5) * (torch.exp(D_dist.log_prob(D0)) + torch.exp(D_dist.log_prob(-D0))))
        
        # if sigmoid
        # FD_D:
        log_prob_D = (D_dist.log_prob(D0) - torch.log(1e-6 + self.duration_max * torch.sigmoid(D0) * (1 - torch.sigmoid(D0)) )).sum(-1)
        return D, log_prob_D

    def update_critic(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)
        
        
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)
        dones_not_max = torch.tensor(data['done_not_max'], dtype=torch.float32)
        # print(1-dones_not_max)
        predictions = self.actor_network(SPs)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 
        log_probs_ZPs = log_probs_ZPs.reshape(-1,1)
        DPs, log_probs_DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)
        log_probs_DPs = log_probs_DPs.reshape(-1,1)

        SPs_ZPs_DPs = torch.cat((SPs,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        
        # log_probs = log_probs_ZPs + log_probs_DPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs_ZPs - self.alpha_D.detach() * log_probs_DPs # TODO like sac add another critic network
        

        target_Qs = Rs + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V 
        current_Qs = self.critic(Ss_Zs_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)
        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        
        predictions = self.actor_network(Ss)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)

        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        # duration loss
        Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        Ds = Ds.reshape(-1,1)
        log_prob_Ds = log_prob_Ds.reshape(-1,1)

        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds)
        
        D_loss = self.alpha_D.detach() * log_prob_Ds - current_Qs_D

        

        # update all
        loss = (z_loss + D_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Zs) - self.target_entropy).detach()).mean()
            alpha_loss += (self.alpha_D *
                            (-(log_prob_Ds) - self.target_entropy_D).detach()).mean()
            self.writer.add_scalar('log_prob_D',-log_prob_Ds.detach().numpy().mean(), global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() )

    def update(self,):
        
        data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        
        self.total_updates += 1
        param = self.config['param']
        # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        c_loss = self.update_critic(data)
        self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
        
        z_loss, D_loss = self.update_actors(data)   
        # z_loss = D_loss=0
        self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        agent_data = {}
        agent_data['critic_loss'] = c_loss
        agent_data['alpha'] = self.alpha.detach().numpy()
        agent_data['alpha_D'] = self.alpha_D.detach().numpy()
        # if self.total_updates % 1000 == 0:
        #     self.test_critic()
        return agent_data
            
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-1, 1, 100)
        D = torch.linspace(0, 1, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S =  0. * Z_
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def alpha_D(self):
        return self.log_alpha_D.exp()

class SAC_async:
    def __init__(self,config, RB_sample_queue, A_Q) -> None:
        # self.writer = config['writer']
        self.config = config
        self.save_time = 0
        # self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        self.rho = 0.4 # its good for 10 sec episodes TODO better explain why.
        param = config['param']
        # critic model inputs are state, z, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        
        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_network = COCT_actor_network_simple(config)
        self.actor_network.share_memory()

        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=param['log_alpha_lr'],)
        # z dim 
        self.target_entropy = -(param['z_dim'])
        
        
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    

        self.RB = RLDataset(D)

        # self.z2omega = lambda x: x
        self.update_process = multiprocessing.Process(target=self.update_process_target, args=(RB_sample_queue,A_Q,config), daemon=True)


    def update_process_target(self,R_sample_Q,A_Q,config):
        counter = 0

        while True:
            counter +=1
            # print('counter',counter)
            while not R_sample_Q.empty():
                sample = R_sample_Q.get()
                # print(sample)
                self.RB.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])
                # print('RB: ', self.RB.real_size)
            # tt = time.time()
            # time.sleep(0.1)

            self.update_async(A_Q,counter) 
    
    def update_async(self,A_Q,counter):
        if self.RB.real_size > 1 * self.config['param']['batch_size'] and self.RB.real_size > 60 / self.config['param']['env_dt']:
            # print('update') 
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
            self.total_updates += 1
            param = self.config['param']
            # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
            agent_data = {}
            agent_data['critic_loss'] = c_loss
            agent_data['alpha'] = self.alpha.detach().numpy()
            if A_Q.empty():
                A_Q.put(agent_data)
        else:
            
            time.sleep(0.01)


    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        
        D = torch.ones_like(mu_D) * self.config['env_dt']
        log_prob_D = torch.zeros_like(mu_D)
        
        return D, log_prob_D

    def update_critic(self, data):

        
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)
        
        
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)
        dones_not_max = torch.tensor(data['done_not_max'], dtype=torch.float32)
        # print(1-dones_not_max)
        predictions = self.actor_network(SPs)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 
        log_probs_ZPs = log_probs_ZPs.reshape(-1,1)
        DPs, log_probs_DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)
        log_probs_DPs = log_probs_DPs.reshape(-1,1)
        SPs_ZPs_DPs = torch.cat((SPs,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        
        log_probs = log_probs_ZPs + log_probs_DPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs # TODO like sac add another critic network
        

        target_Qs = Rs  + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V # TODO add different dones ...
        # target_Qs = Rs + (1 - dones_not_max) * 0.98 *  target_V # TODO add different dones ...        
        current_Qs = self.critic(Ss_Zs_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        
        predictions = self.actor_network(Ss)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)

        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        
        
        

        

        # update all
        # print(z_loss)
        loss = (z_loss).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Zs) - self.target_entropy).detach()).mean()
            # self.writer.add_scalar('alpha',self.alpha, global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        return z_loss.mean().detach().numpy()

    def update(self,):
        param = self.config['param']
        
        data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
        self.total_updates += 1
        param = self.config['param']
        # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        c_loss = self.update_critic(data)
        self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
        
        z_loss = self.update_actors(data)   
        # z_loss = D_loss=0
        self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
        agent_data = {}
        agent_data['critic_loss'] = c_loss
        agent_data['alpha'] = self.alpha.detach().numpy()
            
        return agent_data 
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()


class FiGAR_SAC_async:
    def __init__(self,config, RB_sample_queue, A_Q) -> None:
        # self.writer = config['writer']
        self.config = config
        
        # self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        self.rho = 0.4 # its good for 10 sec episodes TODO better explain why.
        param = config['param']
        # critic model inputs are state, z, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        
        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_network = FiGAR_actor_network(config)
        self.actor_network.share_memory()

        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=param['log_alpha_lr'],)
        # z dim 
        self.target_entropy = -(param['z_dim'])
        
        
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    

        self.RB = RLDataset(D)

        # self.z2omega = lambda x: x
        self.update_process = multiprocessing.Process(target=self.update_process_target, args=(RB_sample_queue,A_Q,config), daemon=True)


    def update_process_target(self,R_sample_Q,A_Q,config):
        counter = 0

        while True:
            counter +=1
            # print('counter',counter)
            while not R_sample_Q.empty():
                sample = R_sample_Q.get()
                # print(sample)
                self.RB.notify(s=sample['s'], sp=sample['sp'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])
                # print('RB: ', self.RB.real_size)
            # tt = time.time()
            # time.sleep(0.1)

            self.update_async(A_Q,counter) 
    
    def update_async(self,A_Q,counter):
        if self.RB.real_size > 1 * self.config['param']['batch_size']:
            # print('update') 
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
            self.total_updates += 1
            param = self.config['param']
            # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_prefs, self.actor_network.actor_D_prefs_target, tau = 0.01)
            agent_data = {}
            agent_data['critic_loss'] = c_loss
            agent_data['alpha'] = self.alpha.detach().numpy()
            if A_Q.empty():
                A_Q.put(agent_data)
        else:
            
            time.sleep(0.01)


    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, prefs_D):
        # print(torch.softmax(prefs_D, dim=-1))
        D_dist = torch.distributions.Categorical(torch.softmax(prefs_D, dim=-1))
        inds = D_dist.sample_n(1) # get length of execution 1 to max
        # D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        # D = D0 ** 2 + self.config['env_dt']
        D = (inds+1) * self.config['env_dt'] # scale to time
        # D = D*0 + 1.0
        # log_prob_D = (D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        # if square of gaussian
        #fD_D:
        # log_prob_D = torch.log(1/(2 * (1.0e-6 + D-self.config['env_dt'])**0.5) * (torch.exp(D_dist.log_prob(D0)) + torch.exp(D_dist.log_prob(-D0))))
        
        # if sigmoid
        # FD_D:
        log_prob_D = 0
        return D, log_prob_D, inds

    def update_critic(self, data):

        
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)
        
        
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)
        dones_not_max = torch.tensor(data['done_not_max'], dtype=torch.float32)
        # print(1-dones_not_max)
        predictions = self.actor_network(SPs)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 
        log_probs_ZPs = log_probs_ZPs.reshape(-1,1)

        DPs, log_probs_DPs, _ =  self.get_duration(predictions['D_prefs_target'])
        # log_probs_DPs = log_probs_DPs.reshape(-1,1)
        DPs = DPs.reshape(-1,1)

        SPs_ZPs_DPs = torch.cat((SPs,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        
        log_probs = log_probs_ZPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs # TODO like sac add another critic network
        

        target_Qs = Rs + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V # TODO add different dones ...
        current_Qs = self.critic(Ss_Zs_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        
        predictions = self.actor_network(Ss)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)


        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        # duration loss
        Ds, log_prob_Ds, inds = self.get_duration(predictions['D_prefs'])
        Ds = Ds.reshape(-1,1)
        
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds)
        # print(torch.softmax(predictions['D_prefs'],dim=1).sum(dim=1))
        D_loss = - current_Qs_D * torch.log_softmax(predictions['D_prefs'],dim=1)[range(len(inds)),inds].reshape(-1,1)
        # print(D_loss.shape)
        

        # update all
        # print(z_loss)
        loss = (z_loss + D_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Zs) - self.target_entropy).detach()).mean()
            # self.writer.add_scalar('alpha',self.alpha, global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() )

    def update(self,):
        param = self.config['param']
        
        data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
        self.total_updates += 1
        param = self.config['param']
        # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        c_loss = self.update_critic(data)
        self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
        
        z_loss, D_loss = self.update_actors(data)   
        # z_loss = D_loss=0
        self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_prefs, self.actor_network.actor_D_prefs_target, tau = 0.01)
        agent_data = {}
        agent_data['critic_loss'] = c_loss
        agent_data['alpha'] = self.alpha.detach().numpy()
            
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()


class visual_COCT_SAC_async:
    def __init__(self,config, RB_sample_queue, A_Q) -> None:
        # self.writer = config['writer']
        ctx = mp.get_context('spawn')
        self.config = config
        self.device = config['device']
        # self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        self.rho = 0.4 # its good for 10 sec episodes TODO better explain why.
        param = config['param']
        if 'duration_max' in param:
            self.duration_max = param['duration_max']
        else:
            self.duration_max = 1.0
        # critic model inputs are state, z, D
        self.critic_encoder = EncoderModel(config['image_shape'], [config['state_dim']], config['net_params'], rad_offset=0)
        self.critic_encoder.to(config['device'])
        config['latent_dim'] = self.critic_encoder.latent_dim

        self.critic = neural_net(   input_size = config['latent_dim'] + param['z_dim'] + 1,
                                    output_size = 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])
        
        self.critic_target = neural_net(   input_size= config['latent_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])
        
        
        
        # self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
        #                         torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
        #                         torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_network = visual_CTCO_network(config)
        self.actor_network.to(config['device'])
        self.critic.to(config['device'])
        self.critic_target.to(config['device'])
        self.actor_network.share_memory()

        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(self.device)
        self.log_alpha_D = torch.tensor(np.log(param['init_alpha'])).to(self.device)
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_D.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha, self.log_alpha_D],
                                                    lr=param['log_alpha_lr'],)
        
        # z dim + duration dim
        self.target_entropy = -param['z_dim']
        self.target_entropy_D = -1
        
        
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+list(self.critic_encoder.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']), 
                Variable('image',config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2]), Variable('image_p', config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2]), 
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    
        print(config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2])
        self.RB = RLDataset(D,n_max_row=int(16e3))

        # self.z2omega = lambda x: x
        self.update_process = ctx.Process(target=self.update_process_target, args=(RB_sample_queue,A_Q,config), daemon=True)


    def update_process_target(self,R_sample_Q,A_Q,config):
        counter = 0

        while True:
            counter +=1
            # print('counter',counter)
            while not R_sample_Q.empty():
                sample = R_sample_Q.get()
                # print(sample)
                
                self.RB.notify(s=sample['s'], image=sample['image'], sp=sample['sp'],image_p=sample['image_p'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])
                # print('RB: ', self.RB.real_size)
            # tt = time.time()
            # time.sleep(0.1)

            self.update_async(A_Q,counter) 
    
    def update_async(self,A_Q,counter):
        if self.RB.real_size > 2 * self.config['param']['batch_size']:
            # print('update') 
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
            self.total_updates += 1
            param = self.config['param']
            # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
            agent_data = {}
            agent_data['critic_loss'] = c_loss
            agent_data['alpha'] = self.alpha.detach().cpu().numpy()
            agent_data['alpha_D'] = self.alpha_D.detach().cpu().numpy()
            
            if A_Q.empty():
                A_Q.put(agent_data)
        else:
            
            time.sleep(0.01)


    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        shift = -0
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        D0 = D_dist.rsample() + shift
        # D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        # D = D0 ** 2 + self.config['env_dt']
        D = self.duration_max * torch.sigmoid(D0) + self.config['env_dt']
        # D = D*0 + 1.0
        # log_prob_D = (D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        # if square of gaussian
        #fD_D:
        # log_prob_D = torch.log(1/(2 * (1.0e-6 + D-self.config['env_dt'])**0.5) * (torch.exp(D_dist.log_prob(D0)) + torch.exp(D_dist.log_prob(-D0))))
        
        # if sigmoid
        # FD_D:
        log_prob_D = (D_dist.log_prob(D0 - shift) - torch.log(1e-6 + self.duration_max * torch.sigmoid(D0) * (1 - torch.sigmoid(D0)) )).sum(-1)
        return D, log_prob_D

    def update_critic(self, data):

        
        Ss = torch.tensor(data['s'], dtype=torch.float32).to(self.device)
        SPs = torch.tensor(data['sp'], dtype=torch.float32).to(self.device)
        Zs = torch.tensor(data['z'], dtype=torch.float32).to(self.device)
        Ds = torch.tensor(data['D'], dtype=torch.float32).to(self.device)
        ds = torch.tensor(data['d'], dtype=torch.float32).to(self.device)
        
        images = torch.tensor(data['image'], dtype=torch.float32).reshape( (data['image'].shape[0],)+ self.config['image_shape'],).to(self.device)
        images_p = torch.tensor(data['image_p'], dtype=torch.float32).reshape((data['image'].shape[0],)+ self.config['image_shape']).to(self.device)
        

        Rs = torch.tensor(data['r'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(data['done'], dtype=torch.float32).to(self.device)
        dones_not_max = torch.tensor(data['done_not_max'], dtype=torch.float32).to(self.device)
        # print(1-dones_not_max)
        predictions = self.actor_network(SPs, images_p)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 

        DPs, log_probs_DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)

        # get latents
        random_rad = False
        detach_encoder = False
        latent_Ps = self.critic_encoder(images_p, SPs, random_rad, detach=detach_encoder)
        latents = self.critic_encoder(images , Ss, random_rad, detach=detach_encoder)
        SPs_ZPs_DPs = torch.cat((latent_Ps,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((latents,Zs.detach(), Ds.detach()), dim=1)
        
        # log_probs = log_probs_ZPs + log_probs_DPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs_ZPs - self.alpha_D.detach() * log_probs_DPs  # TODO like sac add another critic network
        

        target_Qs = Rs + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V # TODO add different dones ...
        current_Qs = self.critic(Ss_Zs_Ds)
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        # current_Vs = self.value(SPs)
        # value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().cpu().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32).to(self.device)
        Ds = torch.tensor(data['D'], dtype=torch.float32).to(self.device)
        images = torch.tensor(data['image'], dtype=torch.float32).reshape( (data['image'].shape[0],)+ self.config['image_shape'],).to(self.device)

        predictions = self.actor_network(Ss, images)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)
        # get critic latents
        random_rad = False
        detach_encoder = False
        latents = self.critic_encoder(images, Ss, random_rad, detach=detach_encoder)

        Ss_Zs_Ds = torch.cat((latents.detach(),Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        # duration loss
        Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        log_prob_Ds = log_prob_Ds.reshape(-1,1)
        Ds = Ds.reshape(-1,1)
        
        Zs = torch.tensor(data['z'], dtype=torch.float32).to(self.device)
        Ss_Zs_Ds = torch.cat((latents.detach(),Zs, Ds), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Ds)
        
        D_loss = self.alpha_D.detach() * log_prob_Ds - current_Qs_D

        

        # update all
        # print(z_loss)
        loss = (z_loss + D_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Zs) - self.target_entropy).detach()).mean()
            alpha_loss += (self.alpha_D *
                            (-(log_prob_Ds) - self.target_entropy_D).detach()).mean()
            # self.writer.add_scalar('alpha',self.alpha, global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        return (z_loss.mean().cpu().detach().numpy() , D_loss.mean().cpu().detach().numpy() )

    def update(self, num_epochs=1):
        param = self.config['param']
        
        for i in range(num_epochs):
            self.total_updates += 1
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        # if self.total_steps % 2000 == 500 and param['log_level']>=1:
        #     D_values = self.RB.get_full()['D'][-500:].reshape(-1)
        #     # print(D_values)
        #     self.writer.add_histogram('Duration hist', D_values, global_step = self.total_steps)
        
        
        # z_loss = D_loss = beta_loss = 0
        # if param['log_level']==2:
        #     self.writer.add_scalars('losses', {'critic':c_loss,
        #                                     'actor_z': z_loss,
        #                                     'actor_D': D_loss,
        #                                     }, self.total_updates, walltime=self.real_t)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    @property
    def alpha_D(self):
        return self.log_alpha_D.exp()


class visual_COCT_SAC_async_eval:
    def __init__(self,config, RB_sample_queue, A_Q) -> None:
        # self.writer = config['writer']
        ctx = mp.get_context('spawn')
        self.config = config
        self.device = config['device']
        # self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        self.rho = 0.4 # its good for 10 sec episodes TODO better explain why.
        param = config['param']
        if 'duration_max' in param:
            self.duration_max = param['duration_max']
        else:
            self.duration_max = 1.0
        # critic model inputs are state, z, D
        self.critic_encoder = EncoderModel(config['image_shape'], [config['state_dim']], config['net_params'], rad_offset=0)
        self.critic_encoder.to(config['device'])
        config['latent_dim'] = self.critic_encoder.latent_dim

        self.critic = neural_net(   input_size = config['latent_dim'] + param['z_dim'] + 1,
                                    output_size = 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])
        
        self.critic_target = neural_net(   input_size= config['latent_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])
        
        
        
        # self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
        #                         torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
        #                         torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_network = visual_CTCO_network(config)
        self.load_actor(config['load_path'])
        
        self.actor_network.to(config['device'])
        self.critic.to(config['device'])
        self.critic_target.to(config['device'])
        self.actor_network.share_memory()

        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(self.device)
        self.log_alpha_D = torch.tensor(np.log(param['init_alpha'])).to(self.device)
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_D.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha, self.log_alpha_D],
                                                    lr=param['log_alpha_lr'],)
        
        # z dim + duration dim
        self.target_entropy = -param['z_dim']
        self.target_entropy_D = -1
        
        
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+list(self.critic_encoder.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']), 
                Variable('image',config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2]), Variable('image_p', config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2]), 
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    
        

    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        shift = -0
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        D0 = D_dist.rsample() + shift
        # D = torch.nn.functional.softplus(D0) + self.config['env_dt']
        # D = D0 ** 2 + self.config['env_dt']
        D = self.duration_max * torch.sigmoid(D0) + self.config['env_dt']
        # D = D*0 + 1.0
        # log_prob_D = (D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        # if square of gaussian
        #fD_D:
        # log_prob_D = torch.log(1/(2 * (1.0e-6 + D-self.config['env_dt'])**0.5) * (torch.exp(D_dist.log_prob(D0)) + torch.exp(D_dist.log_prob(-D0))))
        
        # if sigmoid
        # FD_D:
        log_prob_D = (D_dist.log_prob(D0 - shift) - torch.log(1e-6 + self.duration_max * torch.sigmoid(D0) * (1 - torch.sigmoid(D0)) )).sum(-1)
        return D, log_prob_D

    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    @property
    def alpha_D(self):
        return self.log_alpha_D.exp()



class visual_SAC_async:
    def __init__(self,config, RB_sample_queue, A_Q) -> None:
        # self.writer = config['writer']
        ctx = mp.get_context('spawn')
        self.config = config
        self.device = config['device']
        # self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        self.rho = 0.4 # its good for 10 sec episodes TODO better explain why.
        param = config['param']
        # critic model inputs are state, z, D
        self.critic_encoder = EncoderModel(config['image_shape'], [config['state_dim']], config['net_params'], rad_offset=0)
        self.critic_encoder.to(config['device'])
        config['latent_dim'] = self.critic_encoder.latent_dim

        self.critic = neural_net(   input_size = config['latent_dim'] + param['z_dim'] + 1,
                                    output_size = 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])
        
        self.critic_target = neural_net(   input_size= config['latent_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])
        
        
        
        # self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
        #                         torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
        #                         torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_network = visual_CTCO_network(config)
        self.actor_network.to(config['device'])
        self.critic.to(config['device'])
        self.critic_target.to(config['device'])
        self.actor_network.share_memory()

        self.log_alpha = torch.tensor(np.log(param['init_alpha'])).to(self.device)
        self.log_alpha.requires_grad = param['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=param['log_alpha_lr'],)
        
        # z dim 
        self.target_entropy = -param['z_dim']
        
        
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+list(self.critic_encoder.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']), 
                Variable('image',config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2]), Variable('image_p', config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2]), 
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    
        print(config['image_shape'][0] * config['image_shape'][1] * config['image_shape'][2])
        self.RB = RLDataset(D,n_max_row=int(16e3))

        # self.z2omega = lambda x: x
        self.update_process = ctx.Process(target=self.update_process_target, args=(RB_sample_queue,A_Q,config), daemon=True)


    def update_process_target(self,R_sample_Q,A_Q,config):
        counter = 0

        while True:
            counter +=1
            # print('counter',counter)
            while not R_sample_Q.empty():
                sample = R_sample_Q.get()
                # print(sample)
                
                self.RB.notify(s=sample['s'], image=sample['image'], sp=sample['sp'],image_p=sample['image_p'], r=sample['r'], done_not_max=sample['done_not_max'], done=sample['done'],
                        z=sample['z'], 
                        D=sample['D'], d=sample['d'])
                # print('RB: ', self.RB.real_size)
            # tt = time.time()
            # time.sleep(0.1)

            self.update_async(A_Q,counter) 
    
    def update_async(self,A_Q,counter):
        if self.RB.real_size > 1 * self.config['param']['batch_size']:
            # print('update') 
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            
            self.total_updates += 1
            param = self.config['param']
            # data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
            agent_data = {}
            agent_data['critic_loss'] = c_loss
            agent_data['alpha'] = self.alpha.detach().cpu().numpy()
            
            if A_Q.empty():
                A_Q.put(agent_data)
        else:
            
            time.sleep(0.01)


    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        # print(high, low)
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        D = torch.ones_like(mu_D) * self.config['env_dt']
        log_prob_D = torch.zeros_like(mu_D)
        
        return D, log_prob_D

    def update_critic(self, data):

        
        Ss = torch.tensor(data['s'], dtype=torch.float32).to(self.device)
        SPs = torch.tensor(data['sp'], dtype=torch.float32).to(self.device)
        Zs = torch.tensor(data['z'], dtype=torch.float32).to(self.device)
        Ds = torch.tensor(data['D'], dtype=torch.float32).to(self.device)
        ds = torch.tensor(data['d'], dtype=torch.float32).to(self.device)
        
        images = torch.tensor(data['image'], dtype=torch.float32).reshape( (data['image'].shape[0],)+ self.config['image_shape'],).to(self.device)
        images_p = torch.tensor(data['image_p'], dtype=torch.float32).reshape((data['image'].shape[0],)+ self.config['image_shape']).to(self.device)
        

        Rs = torch.tensor(data['r'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(data['done'], dtype=torch.float32).to(self.device)
        dones_not_max = torch.tensor(data['done_not_max'], dtype=torch.float32).to(self.device)
        # print(1-dones_not_max)
        predictions = self.actor_network(SPs, images_p)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 
        log_probs_ZPs = log_probs_ZPs.reshape(-1,1)
        DPs, log_probs_DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)

        # get latents
        random_rad = False
        detach_encoder = False
        latent_Ps = self.critic_encoder(images_p, SPs, random_rad, detach=detach_encoder)
        latents = self.critic_encoder(images , Ss, random_rad, detach=detach_encoder)
        SPs_ZPs_DPs = torch.cat((latent_Ps,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((latents,Zs.detach(), Ds.detach()), dim=1)
        
        # log_probs = log_probs_ZPs + log_probs_DPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs_ZPs 

        target_Qs = Rs + (1 - dones_not_max) * torch.exp(-self.rho * ds) *  target_V # TODO add different dones ...
        current_Qs = self.critic(Ss_Zs_Ds)
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        # current_Vs = self.value(SPs)
        # value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().cpu().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32).to(self.device)
        Ds = torch.tensor(data['D'], dtype=torch.float32).to(self.device)
        images = torch.tensor(data['image'], dtype=torch.float32).reshape( (data['image'].shape[0],)+ self.config['image_shape'],).to(self.device)

        predictions = self.actor_network(Ss, images)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        log_prob_Zs = log_prob_Zs.reshape(-1,1)

        # get critic latents
        random_rad = False
        detach_encoder = False
        latents = self.critic_encoder(images, Ss, random_rad, detach=detach_encoder)

        Ss_Zs_Ds = torch.cat((latents.detach(),Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        
        

        # update all
        # print(z_loss)
        loss = (z_loss).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()
        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(log_prob_Zs) - self.target_entropy).detach()).mean()
                        
            # self.writer.add_scalar('alpha',self.alpha, global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            # print(self.log_alpha_D.grad, self.log_alpha.grad)  
            self.log_alpha_optimizer.step()

        
        return z_loss.mean().cpu().detach().numpy() 

    def update(self, num_epochs=1):
        param = self.config['param']
        
        for i in range(num_epochs):
            self.total_updates += 1
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        # if self.total_steps % 2000 == 500 and param['log_level']>=1:
        #     D_values = self.RB.get_full()['D'][-500:].reshape(-1)
        #     # print(D_values)
        #     self.writer.add_histogram('Duration hist', D_values, global_step = self.total_steps)
        
        
        # z_loss = D_loss = beta_loss = 0
        # if param['log_level']==2:
        #     self.writer.add_scalars('losses', {'critic':c_loss,
        #                                     'actor_z': z_loss,
        #                                     'actor_D': D_loss,
        #                                     }, self.total_updates, walltime=self.real_t)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    @property
    def alpha_D(self):
        return self.log_alpha_D.exp()



class SAC:
    def __init__(self,config) -> None:
        self.writer = config['writer']
        self.config = config
        
        self.rho = - np.log(config['param']['discount']) / self.config['env_dt']
        param = config['param']
        # critic model inputs are state, z, D
        self.critic = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target = neural_net(   input_size= config['state_dim'] + param['z_dim'] + 1,
                                    output_size= 1,
                                    num_hidden_layer = 2,
                                    hidden_layer_size = param['critic_NN_nhid'],
                                    activation = param['critic_NN_gate'])

        self.critic_target.load_state_dict(self.critic.state_dict())

        
        
        self.value = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] , param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        if param['critic_NN_gate'] == 'ReLU':
            torch.nn.init.kaiming_normal_(self.critic[-1].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-3].weight, nonlinearity='relu') 
            torch.nn.init.kaiming_normal_(self.critic[-5].weight, nonlinearity='relu') 
            self.critic[-1].bias.data[:] = 0*torch.rand(self.critic[-1].bias.data[:].shape)-0
            self.critic[-3].bias.data[:] = 2*torch.rand(self.critic[-3].bias.data[:].shape)-1
            self.critic[-5].bias.data[:] = 2*torch.rand(self.critic[-5].bias.data[:].shape)-1

        self.actor_network = COCT_actor_network_simple(config)
        
        self.log_alpha = torch.tensor(np.log(self.config['init_alpha'])).to(torch.device('cpu'))
        self.log_alpha.requires_grad = self.config['log_alpha_requires_grad']
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.config['log_alpha_lr'],)
        # z dim + duration dim
        self.target_entropy = -(param['z_dim']+1)
        
        
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']),
                Variable('d',1),
                Variable('D',1),
                Variable('r',1), Variable('done',1), Variable('done_not_max',1))    
    

        self.RB = RLDataset(D)
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.real_t = 0.
        


        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])


        self.z2omega = lambda x: x

    def get_action_z(self, mu_z, sigma_z):
        
        z_dist = torch.distributions.Normal(mu_z, sigma_z+1e-4)
        z0 = z_dist.rsample()
        z = self.scale_action(z0)
        high = self.config['action_high']
        low = self.config['action_low']
        
        k = torch.tensor((high-low)/2)
        # print(z0.shape, z.shape)
        log_prob_z = (z_dist.log_prob(z0) - torch.log( k * (1 - torch.tanh(z0).pow(2) + 1e-6) )).sum(-1)
        
        return z, log_prob_z
        
    def get_duration(self, mu_D, sigma_D):
        D_dist = torch.distributions.Normal(mu_D, sigma_D+1e-4)
        D0 = D_dist.rsample()
        # force to dt
        D = torch.ones_like(D0) * self.config['env_dt']
        log_prob_D = 0*(D_dist.log_prob(D0) - torch.log( 1 - torch.sigmoid(-D) + 1e-6)).sum(-1)
        
        return D, log_prob_D

    def update_critic(self, data):

        
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)

        
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)

        predictions = self.actor_network(SPs)
        
        
        ZPs, log_probs_ZPs =  self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 

        DPs, log_probs_DPs =  self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4)
        DPs = DPs.reshape(-1,1)

        SPs_ZPs_DPs = torch.cat((SPs,ZPs, DPs), dim=1)
        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        
        log_probs = log_probs_ZPs + log_probs_DPs
        # print(log_probs_ZPs )
        target_V = self.critic_target(SPs_ZPs_DPs) - self.alpha.detach() * log_probs # TODO like sac add another critic network
        

        target_Qs = Rs + torch.exp(-self.rho * ds) *  target_V # TODO add different dones ...
        current_Qs = self.critic(Ss_Zs_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_DPs).detach())**2).mean()
        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):

        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        
        predictions = self.actor_network(Ss)

        
        
        # high-policy loss
        Zs, log_prob_Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
       

        Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Ds)
        z_loss = self.alpha.detach() * log_prob_Zs - current_Qs_z
               
        # duration loss
        # Ds, log_prob_Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        # Ds = Ds.reshape(-1,1)
        
        # Zs = torch.tensor(data['z'], dtype=torch.float32)
        # Ss_Zs_Ds = torch.cat((Ss,Zs, Ds), dim=1)
        # current_Qs_D = self.critic(Ss_Zs_Ds)
        
        # D_loss = self.alpha.detach() * log_prob_Ds - current_Qs_D

        

        # update all
        # print(z_loss)
        loss = (z_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        
        loss.backward()
        self.actor_network.actor_optimizer.step()

        if self.log_alpha.requires_grad:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                            (-(0+log_prob_Zs) - self.target_entropy).detach()).mean()
            self.writer.add_scalar('alpha',self.alpha, global_step=self.total_updates)
        # alpha_loss = (self.alpha *
        #                 (entropy - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        return (z_loss.mean().detach().numpy() , 0.0 )

    def update(self, num_epochs=1):
        param = self.config['param']
        
        for i in range(num_epochs):
            self.total_updates += 1
            data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
            c_loss = self.update_critic(data)
            self.soft_update_params(self.critic, self.critic_target, tau = param['critic_target_update_tau'])
            
            z_loss, D_loss = self.update_actors(data)   
            # z_loss = D_loss=0
            self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
            self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        if self.total_steps % 2000 == 500 and param['log_level']>=1:
            D_values = self.RB.get_full()['D'][-500:].reshape(-1)
            # print(D_values)
            self.writer.add_histogram('Duration hist', D_values, global_step = self.total_steps)
        
        
        # z_loss = D_loss = beta_loss = 0
        if param['log_level']==2:
            self.writer.add_scalars('losses', {'critic':c_loss,
                                            'actor_z': z_loss,
                                            'actor_D': D_loss,
                                            }, self.total_updates, walltime=self.real_t)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def test_critic(self):
        Z = torch.linspace(-2, 2, 100)
        D = torch.linspace(0, 2, 100)
        
        Z_,D_ = torch.meshgrid([Z,D])
        Z_ = Z_.unsqueeze(-1)
        D_ = D_.unsqueeze(-1)
        
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,D_.shape)
        SZD = torch.cat((S,Z_,D_),dim=-1)
        Y = self.critic(SZD).squeeze(-1)
        Z_ = Z_.squeeze(-1)
        D_ = D_.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Z_.detach().numpy(), D_.detach().numpy(), Y.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Z_.detach().numpy().min(), Z_.detach().numpy().max(), D_.detach().numpy().min(), D_.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('critic', fig, global_step=0*self.total_steps)

        # plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        # plt.savefig('value_{}.png'.format(ID))
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)

        plt.close()

        
    def test_value(self):
        # test for pendulum
        theta = torch.linspace(-torch.pi, torch.pi, 100)
        theta_dot = torch.linspace(-8., 8., 100)
        
        Theta, Theta_dot = torch.meshgrid([theta, theta_dot])
        Theta = Theta.unsqueeze(-1)
        Theta_dot = Theta_dot.unsqueeze(-1)
        cos_theta = torch.cos(Theta)
        sin_theta = torch.sin(Theta)
        X = torch.cat([cos_theta, sin_theta, Theta_dot],dim=-1)
        Z = self.value(X).squeeze(-1)
        Theta = Theta.squeeze(-1)
        Theta_dot = Theta_dot.squeeze(-1)
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy(),
        #         cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.imshow(Z.detach().numpy(), cmap='viridis')
        plt.pcolormesh(Theta.detach().numpy(), Theta_dot.detach().numpy(), Z.detach().numpy())# cmap='RdBu', vmin=Z.detach().numpy().min, vmax=Z.detach().numpy().max())
        plt.colorbar()
        plt.axis([Theta.detach().numpy().min(), Theta.detach().numpy().max(), Theta_dot.detach().numpy().min(), Theta_dot.detach().numpy().max()])
        # plt.colorbar()
        # ax.set_title('surface')
        self.writer.add_figure('value', fig, global_step=0*self.total_steps)
        plt.close()

    def plot_policy(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('policy.png')
        self.writer.add_figure('policy', fig, global_step=0*self.total_steps)

        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.actor_network(States)
        Y, _ = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        # plt.savefig('duration.png')
        self.writer.add_figure('duration', fig, global_step=0*self.total_steps)

        plt.close()
        
    def scale_action(self,a):
        high = self.config['action_high']
        low = self.config['action_low']
        k = torch.tensor((high-low)/2)
        b = torch.tensor((high+low)/2)
        return k * torch.tanh(a) + b

    def load_actor(self, filename):
        self.actor_network.load_state_dict(torch.load(filename))
        
    def save_actor(self, filename):
        # saving whole model like baseagent
        torch.save(self.actor_network.state_dict(),filename)

    @property
    def alpha(self):
        return self.log_alpha.exp()
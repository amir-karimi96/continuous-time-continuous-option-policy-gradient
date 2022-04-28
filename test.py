# actor critic continuous option continuous time 
import torch 
import numpy as np
from env_wrapper import  D2C, Env_test
import gym
import argparse
import yaml
import matplotlib.pyplot as plt
from dataset import Variable, Domain, RLDataset
from torch.utils.tensorboard import SummaryWriter
import time
from mpl_toolkits import mplot3d
from matplotlib import cm

# Writer will output to ./runs/ directory by default
parser = argparse.ArgumentParser()
parser.add_argument('--ID',default=0,type=int, help='param ID ')
parser.add_argument('--config',default='0',type=str, help='config file name')
parser.add_argument('--result_path',default='0',type=str, help='result folder path')
args = parser.parse_args()
ID = args.ID
writer = SummaryWriter(log_dir='results_{}/'.format(ID))

def neural_net(input_size, output_size, num_hidden_layer, hidden_layer_size, activation):
    NN = torch.nn.Sequential()
    if num_hidden_layer > 0:
        NN.append(torch.nn.Linear(input_size, hidden_layer_size))
        NN.append(getattr(torch.nn, activation)())
        for i in range(num_hidden_layer-1):
            NN.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            NN.append(getattr(torch.nn, activation)())
        NN.append(torch.nn.Linear(hidden_layer_size, output_size))
    else:
        NN.append(torch.nn.Linear(input_size, output_size))
    return NN

class COCT_actor_network(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.std_depend_on_state = param['std_depend_on_state']
        
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        self.actor_z_mu = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma.append(torch.nn.Softplus())

        self.actor_D_sigma = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma.append(torch.nn.Softplus())

        self.actor_z_mu_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_D_mu_target = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        
        self.actor_z_sigma_target = neural_net(input_size= config['state_dim'], output_size= param['z_dim'], num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_z_sigma_target.append(torch.nn.Softplus())

        self.actor_D_sigma_target = neural_net(input_size= config['state_dim'], output_size= 1, num_hidden_layer=2, hidden_layer_size= param['actor_NN_nhid'], activation= param['actor_NN_gate'])
        self.actor_D_sigma_target.append(torch.nn.Softplus())
        
        self.actor_z_mu_target.load_state_dict(self.actor_z_mu.state_dict())
        self.actor_D_mu_target.load_state_dict(self.actor_D_mu.state_dict())
        self.actor_z_sigma_target.load_state_dict(self.actor_z_sigma.state_dict())
        self.actor_D_sigma_target.load_state_dict(self.actor_D_sigma.state_dict())
        
        # option terminaiton model beta(T|s,z_{-1})
        self.beta = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] + param['z_dim'], param['beta_NN_nhid']), getattr(torch.nn, param['beta_NN_gate'])(),
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

    def forward(self,state, z_):
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
        
        predictions = { 'beta': 0.*self.beta(torch.cat((state, z_), dim=-1)),
                        'z_mu': z_mu ,
                        'z_sigma': z_sigma ,
                        'D_mu':   D_mu - 1.5 ,
                        'D_sigma':  D_sigma / 50,
                        'z_mu_target': z_mu_target,
                        'z_sigma_target': z_sigma_target ,
                        'D_mu_target': D_mu_target -1.5,
                        'D_sigma_target': D_sigma_target / 50,
                        # 'omega': self.z2omega( z_D[...,:-1] ),
                        }
        return predictions

class COCT:
    def __init__(self,config) -> None:
        print(config['state_dim'] ) 
        self.config = config
        self.dt = config['param']['dt']
        self.rho = - np.log(config['param']['discount']) / self.dt
        
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

        self.actor_network = COCT_actor_network(config)
        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        
        
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']), Variable('z_prev',param['z_dim']),
                Variable('T', 1),
                Variable('tau',1), Variable('tau_prev',1),
                Variable('D',1), Variable('D_prev',1),
                Variable('d',1), Variable('r',1), Variable('done',1))    
    

        self.RB = RLDataset(D)
        self.total_steps = 0
        self.real_t = 0.
        if config['load'] :
            #load
            pass

        
        # option terminaiton model beta(T|s,z_{-1})
        self.beta = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] + param['z_dim'], param['beta_NN_nhid']), getattr(torch.nn, param['beta_NN_gate'])(),
                                torch.nn.Linear(param['beta_NN_nhid'], param['beta_NN_nhid']), getattr(torch.nn, param['beta_NN_gate'])(),
                                torch.nn.Linear(param['beta_NN_nhid'], 1),torch.nn.Sigmoid())

        # self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['critic_lr'])

        # self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])


        self.z2omega = lambda x: x

    def get_action_z(self, mu_z, sigma_z):
        z = torch.distributions.Normal(mu_z, sigma_z+1e-4).rsample()
        z = self.scale_action(z)
        return z
    def get_duration(self, mu_D, sigma_D):
        D = torch.distributions.Normal(mu_D, sigma_D+1e-4).rsample()
        D = torch.nn.functional.softplus(D)+ continuous_env.dt - 0.01
        return D

    def update_critic(self, data):
        
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)

        Taus = torch.tensor(data['tau'], dtype=torch.float32)
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)

        predictions = agent.actor_network(SPs,Zs)
        
        # sample terminations
        TPs = torch.distributions.Categorical(torch.cat([1-predictions['beta'], predictions['beta']], dim=-1)).sample().reshape(-1,1)
        # determine newmovement
        new_movementPs = ((Taus + self.dt) >= Ds)

        new_movementPs = new_movementPs + (TPs==1) # newmovement or T
        new_movementPs = new_movementPs * 1.
        # print(new_movementPs)
    
        ZPs = Zs * (new_movementPs==0) + (new_movementPs==1) * self.get_action_z(predictions['z_mu_target'], predictions['z_sigma_target']+1e-4) 

        DPs = Ds * (new_movementPs==0) + (new_movementPs==1) * self.get_duration(predictions['D_mu_target'], predictions['D_sigma_target']+1e-4).reshape(-1,1)

        TauPs = (Taus + ds) * (new_movementPs==0) + 0 * (new_movementPs==1)
        SPs_ZPs_TauPs_DPs = torch.cat((SPs,ZPs,TauPs, DPs), dim=1)
        Ss_Zs_Taus_Ds = torch.cat((Ss,Zs,Taus, Ds), dim=1)
        
        target_Qs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.critic_target(SPs_ZPs_TauPs_DPs)
        current_Qs = self.critic(Ss_Zs_Taus_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.critic_target(SPs_ZPs_TauPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Zs_prev = torch.tensor(data['z_prev'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ds_prev = torch.tensor(data['D_prev'], dtype=torch.float32)
        Taus_prev = torch.tensor(data['tau_prev'], dtype=torch.float32)

        predictions = agent.actor_network(Ss,Zs_prev)

        q = 1*(Taus_prev + self.dt < Ds_prev)
        # print(q)
        k =  (1-q) + q * predictions['beta'].detach()
        
        # high-policy loss
        Zs = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
       

        Ss_Zs_Taus_Ds = torch.cat((Ss,Zs, 0*Ds, Ds), dim=1)
        current_Qs_z = self.critic(Ss_Zs_Taus_Ds)
        z_loss = -  k * current_Qs_z
               
        # duration loss
        Ds = self.get_duration(predictions['D_mu'], predictions['D_sigma']).reshape(-1,1)
        
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ss_Zs_Taus_Ds = torch.cat((Ss,Zs,(0*Ds).detach(), Ds), dim=1)
        current_Qs_D = self.critic(Ss_Zs_Taus_Ds)
        D_loss = - k * current_Qs_D

        # beta loss
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ss_Zs_Taus_Ds = torch.cat((Ss,Zs, 0*Ds, Ds), dim=1)
        Qs = self.critic(Ss_Zs_Taus_Ds)
        Ss_Zs__Taus_Ds_ = torch.cat((Ss, Zs_prev, Taus_prev + self.dt, Ds_prev), dim=1)
        Qs_ = self.critic(Ss_Zs__Taus_Ds_)
        beta_loss = - q * predictions['beta'] * (Qs - Qs_).detach()

        # update all
        # print(z_loss)
        loss = (z_loss + D_loss + beta_loss ).mean()

        self.actor_network.actor_optimizer.zero_grad()
        # self.beta_opt.zero_grad()
        loss.backward()
        self.actor_network.actor_optimizer.step()
        # self.beta_opt.step()
        
        writer.add_scalars('z_D_batch_size', {'critic':(1-q).sum().detach(),
                                                }, self.total_steps, walltime=self.real_t)
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() , beta_loss.mean().detach().numpy() )

    def update(self):
        data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        
        c_loss = self.update_critic(data)
       
        self.soft_update_params(self.critic, self.critic_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_z_mu, self.actor_network.actor_z_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_z_sigma, self.actor_network.actor_z_sigma_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_mu, self.actor_network.actor_D_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_network.actor_D_sigma, self.actor_network.actor_D_sigma_target, tau = 0.01)
        
        if self.total_steps % 10000 == 5000:
            D_values = self.RB.get_full()['D'][-5000:].reshape(-1)
            # print(D_values)
            writer.add_histogram('Duration hist', D_values, global_step = self.total_steps)
        
        if self.total_steps % 2 == 0 :#and self.total_steps > 1000:
            z_loss, D_loss, beta_loss = self.update_actors(data)
            # z_loss = D_loss = beta_loss = 0
            writer.add_scalars('losses', {'critic':c_loss,
                                                'actor_z': z_loss,
                                                'actor_D': D_loss,
                                                'beta': beta_loss}, self.total_steps, walltime=self.real_t)

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
        T = 0. * D_
        S = torch.cat([ 0. * Z_, 1. + 0. * Z_, 0. * Z_],dim=-1)
        print(S.shape,Z_.shape,T.shape,D_.shape)
        SZTD = torch.cat((S,Z_,T,D_),dim=-1)
        Y = self.critic(SZTD).squeeze(-1)
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
        writer.add_figure('critic', fig, global_step=0*self.total_steps)

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
        plt.savefig('value_{}.png'.format(ID))
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
        return 2 * torch.tanh(a)

def evaluate():
    pass

if __name__ == '__main__':

    fig, axs = plt.subplots(3)

    
    

    run_ID = args.ID
    cfg_path = args.config
    print(cfg_path)
    result_path = args.result_path
    with open(cfg_path) as file:
        config = yaml.full_load(file)

    param = config['param']
    
    # env = gym.make(param['env'])
    env = Env_test()
    state_dim = len(env.observation_space.sample())
    action_dim = len(env.action_space.sample())
    config['state_dim'] = state_dim
    z_dim = param['z_dim']
    # create dataset
    
    
    RB_list = []

    agent = COCT(config)
    

    num_ep = 10000
    continuous_env = D2C(discrete_env= env, low_level_funciton= lambda x,y,z: x, rho = agent.rho)
    t = 0.
    Returns = []
    for e in range(num_ep):
        s = continuous_env.reset()
        S = torch.tensor(s, dtype = torch.float32)
        z_prev = torch.zeros((param['z_dim'],))
        tau = 0.
        tau_prev = 0.
        D_prev = torch.tensor([0.])

        new_movement = 1
        r_newmovement = 0
        undiscounted_rewards = []
        durations = []
        while True :

            # get actor net outputs based on current state and previous option
            predictions = agent.actor_network(S,z_prev)
            
            # sample termination
            T = torch.distributions.Categorical(torch.tensor([1-predictions['beta'], predictions['beta']])).sample()
            
            if new_movement or T :
                # sample z
                z = agent.get_action_z(predictions['z_mu'], predictions['z_sigma'])
                
                # sample duration 
                D = agent.get_duration(predictions['D_mu'], predictions['D_sigma'])
                tau = 0.
                r_newmovement = -1 * 0.05
                new_movement = False
                # sample movement omega
                omega = agent.z2omega(z) # TO DO add stochasticity

            d = min((D-tau).detach().numpy()[0], agent.dt)
            
            sp,R,done,info = continuous_env.step(omega.detach().numpy(), d)
            # print('R: ', R )
            # print((np.array(info['rewards']) * np.array(info['durations'])).sum())
            # R -= 0.03/D.detach().numpy()
            # R += r_newmovement
            r_newmovement = 0
            agent.RB.notify(s=S.detach().numpy(), sp=sp, r=R, done=done, T=T,
                        z=z.detach().numpy(), z_prev=z_prev.detach().numpy(), 
                        tau=tau, tau_prev=tau_prev, D=D.detach().numpy(), D_prev=D_prev.detach().numpy(), d=d)
            
            undiscounted_rewards.extend(info['rewards'])
            durations.extend(info['durations'])
            # RB_list.append([s, sp, z, R, done, D, T, tau, agent.real_t])
            # print('R: ', R)
            agent.total_steps += 1
            
            writer.add_scalars('timings', {'T':T.detach().numpy(),
                                            'D': D.detach().numpy(),
                                            'D_sigma': predictions['D_sigma'].detach().numpy(),
                                            'tau': tau,
                                            }, agent.total_steps, walltime=agent.real_t)
            writer.add_scalars('action', {'z_mu': z,
                                            'z_sigma': predictions['z_sigma_target'][0],
                                            }, agent.total_steps, walltime=agent.real_t)
            writer.add_scalar('Reward', R, agent.total_steps)
            if agent.total_steps % 1000 == 0 :
                # agent.test_critic()
                agent.plot_value()
                agent.plot_policy()
                agent.plot_duration()
                
            if agent.RB.real_size > 1 * config['param']['batch_size']:
                agent.update()

            if done:
                Returns.append((np.array(undiscounted_rewards) * np.array(durations)).sum())
                # print(np.array(durations).sum())
                # print(undiscounted_rewards)
                # print(R)
                writer.add_scalar('Return_discrete', Returns[-1], e)
                break
            
            z_prev = z
            D_prev = D
            tau_prev = tau
            tau += d
            agent.real_t += d
            S = torch.tensor(sp, dtype=torch.float32)
            # print(tau, D)
            if tau >= D:
                new_movement = True
            
            
        # RB = np.array(RB)
        # print(torch.tensor([k[-1] for k in RB]).detach().numpy())
        # print(RB.get_minibatch(size = 2))
    # ts = [k[-1] for k in RB_list]
    # taus = [k[-2] for k in RB_list]
    # Ts = torch.tensor([k[-3] for k in RB_list]).detach().numpy()
    # Ds = torch.tensor([k[-4] for k in RB_list]).detach().numpy()
    # axs[0].plot(ts, taus)#,marker='-')
    # axs[0].set_ylabel('tau')
    # axs[2].plot(ts, Ts)#,marker='o')
    # axs[2].set_ylabel('Termination')
    # axs[1].plot(ts, Ds)#,marker='o')
    # axs[1].set_ylabel('Duration')
    # plt.show()

writer.close()
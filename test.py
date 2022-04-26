# actor critic continuous option continuous time 
from typing_extensions import Self
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

class COCT_network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(state, z_):
        pass

class COCT:
    def __init__(self,config) -> None:
        print(config['state_dim'] ) 
        self.config = config
        self.dt = config['param']['dt']
        self.rho = - np.log(config['param']['discount']) / self.dt
        # critic model inputs are state, z, tau, D
        self.critic = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] + param['z_dim'] + 1 + 1, param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        self.target_critic = torch.nn.Sequential(torch.nn.Linear(config['state_dim'] + param['z_dim'] + 1 + 1, param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], param['critic_NN_nhid']), getattr(torch.nn, param['critic_NN_gate'])(),
                                torch.nn.Linear(param['critic_NN_nhid'], 1))

        self.target_critic.load_state_dict(self.critic.state_dict())

        
        
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

        # continuous option (actor) mu(z|s) and Duration delta(D|s)
        # can be loaded from imitation learning result
        self.actor_z_D_mu = torch.nn.Sequential(torch.nn.Linear(config['state_dim'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['z_dim']+1))
        self.actor_z_D_sigma = torch.nn.Sequential(torch.nn.Linear(config['state_dim'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['z_dim']+1), torch.nn.Softplus())
        
        self.actor_z_D_mu_target = torch.nn.Sequential(torch.nn.Linear(config['state_dim'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['z_dim']+1))
        
        self.actor_z_D_sigma_target = torch.nn.Sequential(torch.nn.Linear(config['state_dim'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['actor_NN_nhid']), getattr(torch.nn, param['actor_NN_gate'])(),
                                torch.nn.Linear(param['actor_NN_nhid'], param['z_dim']+1), torch.nn.Softplus())
        
        D = Domain(Variable('s',config['state_dim']), Variable('sp',config['state_dim']),
                Variable('z',param['z_dim']), Variable('z_',param['z_dim']),
                Variable('T', 1),
                Variable('tau',1), Variable('tau_',1),
                Variable('D',1), Variable('D_',1),
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

        self.actor_opt = getattr(torch.optim, param['actor_opt'])(list(self.actor_z_D_mu.parameters())+list(self.actor_z_D_sigma.parameters()) ,lr=param['actor_lr'])

        self.critic_opt = getattr(torch.optim, param['critic_opt'])(list(self.critic.parameters())+ list(self.value.parameters()) ,lr=param['actor_lr'])

        self.beta_opt = getattr(torch.optim, param['beta_opt'])(list(self.beta.parameters()) ,lr=param['actor_lr'])


        self.z2omega = lambda x: x

    def get_action_z(self, mu_z, sigma_z):
        z = torch.distributions.Normal(mu_z, sigma_z+1e-4).rsample()
        z = self.scale_action(z)
        return z
    def get_duration(self, mu_D, sigma_D):
        D = torch.distributions.Normal(mu_D, sigma_D+1e-4).rsample()
        D = torch.nn.functional.softplus(D)+ continuous_env.dt
        return D

    def step(self, state, z_):
        # inputs:   current state
        #           previous option z_
        z_D_mu= self.actor_z_D_mu(state)
        z_D_sigma = self.actor_z_D_sigma(state)
        z_D_mu_target= self.actor_z_D_mu(state)
        z_D_sigma_target = self.actor_z_D_sigma(state)
        predictions = { 'beta': 0.*self.beta(torch.cat((state, z_), dim=-1)),
                        'z_mu': z_D_mu[...,:-1],
                        'z_sigma': z_D_sigma[...,:-1],
                        'D_mu':   z_D_mu[...,-1],
                        'D_sigma':  z_D_sigma[..., -1],
                        'z_mu_target': z_D_mu_target[...,:-1],
                        'z_sigma_target': z_D_sigma_target[...,:-1],
                        'D_mu_target': z_D_mu_target[...,-1],
                        'D_sigma_target': z_D_sigma_target[..., -1],
                        # 'omega': self.z2omega( z_D[...,:-1] ),
                        }
        return predictions

    def update_critic(self, data):
        
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        SPs = torch.tensor(data['sp'], dtype=torch.float32)
        Zs = torch.tensor(data['z'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        ds = torch.tensor(data['d'], dtype=torch.float32)

        Taus = torch.tensor(data['tau'], dtype=torch.float32)
        Rs = torch.tensor(data['r'], dtype=torch.float32)
        dones = torch.tensor(data['done'], dtype=torch.float32)

        predictions = agent.step(SPs,Zs)
        
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
        
        target_Qs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.target_critic(SPs_ZPs_TauPs_DPs)
        current_Qs = self.critic(Ss_Zs_Taus_Ds.detach())
        # target_Vs = Rs + torch.exp(-self.rho * ds) * (1-dones) * self.value(SPs)

        current_Vs = self.value(SPs)
        value_loss = ((current_Vs - self.target_critic(SPs_ZPs_TauPs_DPs).detach())**2).mean()

        critic_loss = ((current_Qs - target_Qs.detach())**2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        value_loss.backward()
        self.critic_opt.step()
        return critic_loss.detach().numpy()

    def update_actors(self, data):
        Ss = torch.tensor(data['s'], dtype=torch.float32)
        Zs_ = torch.tensor(data['z_'], dtype=torch.float32)
        Ds = torch.tensor(data['D'], dtype=torch.float32)
        Ds_ = torch.tensor(data['D_'], dtype=torch.float32)
        Taus_ = torch.tensor(data['tau_'], dtype=torch.float32)

        predictions = agent.step(Ss,Zs_)

        q = 1*(Taus_ + self.dt < Ds_)
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
        Ss_Zs__Taus_Ds_ = torch.cat((Ss, Zs_, Taus_ + self.dt, Ds_), dim=1)
        Qs_ = self.critic(Ss_Zs__Taus_Ds_)
        beta_loss = - q * predictions['beta'] * (Qs - Qs_).detach()

        # update all
        loss = (z_loss + D_loss + beta_loss ).mean()

        self.actor_opt.zero_grad()
        # self.beta_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        # self.beta_opt.step()
        writer.add_scalars('z_D_batch_size', {'critic':(1-q).sum().detach(),
                                                }, self.total_steps, walltime=self.real_t)
        return (z_loss.mean().detach().numpy() , D_loss.mean().detach().numpy() , beta_loss.mean().detach().numpy() )

    def update(self):
        data = self.RB.get_minibatch(size = self.config['param']['batch_size'])
        
        c_loss = self.update_critic(data)
       
        self.soft_update_params(self.critic, self.target_critic, tau = 0.01)
        self.soft_update_params(self.actor_z_D_mu, self.actor_z_D_mu_target, tau = 0.01)
        self.soft_update_params(self.actor_z_D_sigma, self.actor_z_D_sigma_target, tau = 0.01)
        
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
        plt.savefig('critic_{}.png'.format(ID))
        plt.close()

    def plot_value(self):
        X = torch.linspace(-10,10,100).reshape(-1,1)
        Y = self.value(X)
        fig = plt.figure()
        plt.plot(X.detach().numpy(), Y.detach().numpy())
        plt.savefig('value_{}.png'.format(ID))
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
        predictions = self.step(States, States*0)
        Y = self.get_action_z(predictions['z_mu'], predictions['z_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        plt.savefig('policy.png')
        plt.close()
    
    def plot_duration(self):
        States = torch.linspace(-2.5, 2.5,100).reshape(-1,1)
        predictions = self.step(States, States*0)
        Y = self.get_duration(predictions['D_mu'], predictions['D_sigma'])
        fig = plt.figure()
        plt.plot(States.detach().numpy(), Y.detach().numpy())
        plt.savefig('duration.png')
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
        z_ = torch.zeros((param['z_dim'],))
        tau = 0.
        tau_ = 0.
        D_ = torch.tensor([0.])

        new_movement = 1
        r_newmovement = 0
        undiscounted_rewards = []
        while True :
            predictions = agent.step(S,z_)
            # sample termination
            T = torch.distributions.Categorical(torch.tensor([1-predictions['beta'], predictions['beta']])).sample()
            
            if new_movement or T :
                # sample z
                z = agent.get_action_z(predictions['z_mu'], predictions['z_sigma'])
                # sample duration 
                # should be positive
                D= agent.get_duration(predictions['D_mu'], predictions['D_sigma'])
                # print(D)
                tau = 0.
                r_newmovement = -1 * 0.05
                new_movement = False
                # sample movement omega
                omega = agent.z2omega(z) # TO DO add stochasticity

            d = min((D-tau).detach().numpy(), agent.dt)
            
            sp,R,done,info = continuous_env.step(omega.detach().numpy(), d)
            # R -= 0.03/D.detach().numpy()
            # R += r_newmovement
            r_newmovement = 0
            agent.RB.notify(s=S.detach().numpy(), sp=sp, r=R, done=done, T=T,
                        z=z.detach().numpy(), z_=z_.detach().numpy(), 
                        tau=tau, tau_=tau_, D=D.detach().numpy(), D_=D_.detach().numpy(), d=d)
            
            undiscounted_rewards.extend(info['rewards'])
            # RB_list.append([s, sp, z, R, done, D, T, tau, agent.real_t])
            
            agent.total_steps += 1
            
            writer.add_scalars('timings', {'T':T.detach().numpy(),
                                            'D': D.detach().numpy(),
                                            'tau': tau,
                                            }, agent.total_steps, walltime=agent.real_t)
            writer.add_scalars('action', {'z_mu': predictions['z_mu_target'][0],
                                            'z_sigma': predictions['z_sigma_target'][0]}, agent.total_steps, walltime=agent.real_t)
            if agent.total_steps % 1000 == 0 :
                # agent.test_critic()
                agent.plot_value()
                agent.plot_policy()
                agent.plot_duration()
            if agent.RB.real_size > 1 * config['param']['batch_size']:
                agent.update()

            if done:
                Returns.append(np.array(undiscounted_rewards).sum())
                # print(undiscounted_rewards)
                # print(R)
                writer.add_scalar('Return_discrete', Returns[-1], e)
                break
            
            z_ = z
            D_ = D
            tau_ = tau
            tau += d
            agent.real_t += d
            S = torch.tensor(sp, dtype=torch.float32)
            
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
import numpy as np
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control import Continuous_MountainCarEnv
import matplotlib.pyplot as plt
import gym
from gym.envs.mujoco import HalfCheetahEnv, HopperEnv
from dm_control import suite
from dm_control.suite import ball_in_cup
from sub_policies import sub_policy
try:
    import franka_gym



    import gym
    from dm_control import suite
    from dm_control.suite import ball_in_cup
except:
    pass
class CT_dm:
    def __init__(self, dt = 0.01) -> None:
        self.env = suite.load(domain_name="ball_in_cup", task_name="catch", environment_kwargs={'control_timestep': dt})
        
        self.observation_space = spaces.Box(low=-2., high=2., shape=(8,), dtype=np.float32)
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(low=action_spec.minimum, high=action_spec.maximum, shape=action_spec.shape, dtype=np.float32)

        self.DT = dt
        self.dt = dt
        self.episode_length = int(20 / dt) 
        self.counter = 0

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        # self.DT = 0.04
        self.dt = self.DT
        time_step = self.env.reset()
        # time_step.reward, time_step.discount, time_step.observation
        obs = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
        
        return obs

    def step(self, action,d=None):
        self.counter +=1 
        # action[-1] = 1
        # print(action)
        time_step = self.env.step(action)
        # time_step.reward, time_step.discount, time_step.observation
        # print(time_step.observation)
        obs = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
        reward = time_step.reward
        info = {}
        done = time_step.last()
        # reward = reward / self.DT
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return obs, reward, done, info
    def seed(self, i):
        pass
    def render(self, mode):
        return self.env.physics.render()

class CT_hopper_dm:
    def __init__(self, dt = 0.01) -> None:
        self.env = suite.load(domain_name="hopper", task_name="hop", environment_kwargs={'control_timestep': dt})
        #self.env = suite.load(domain_name="hopper", task_name="hop", task_kwargs={'random':0}, environment_kwargs={'control_timestep': dt})
        self.observation_space = spaces.Box(low=-2., high=2., shape=(15,), dtype=np.float32)
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(low=action_spec.minimum, high=action_spec.maximum, shape=action_spec.shape, dtype=np.float32)

        self.DT = dt
        self.dt = dt
        self.episode_length = int(20 / dt) 
        self.counter = 0
        print(self.episode_length)

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        # self.DT = 0.04
        self.dt = self.DT
        time_step = self.env.reset()
        # time_step.reward, time_step.discount, time_step.observation
        obs = np.concatenate((time_step.observation['position'], time_step.observation['velocity'], time_step.observation['touch']))

        return obs

    def step(self, action,d=None):
        self.counter +=1 
        # action[-1] = 1
        # print(action)
        time_step = self.env.step(action)
        # time_step.reward, time_step.discount, time_step.observation
        # print(time_step.observation)
        obs = np.concatenate((time_step.observation['position'], time_step.observation['velocity'], time_step.observation['touch']))
        reward = time_step.reward
        info = {}
        done = time_step.last()
        # reward = reward / self.DT
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return obs, reward, done, info
    def seed(self,i):
        pass
    def render(self, mode):
        return self.env.physics.render()


class CT_half_cheetah:
    def __init__(self, dt=0.05) -> None:
        self.env = HalfCheetahEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        # self.dt = dt
        self.DT = dt
        self.dt = self.DT
        self.env.model.opt.timestep = dt / self.env.frame_skip
        self.counter = 0
        self.max_episode_length = int(1000 * 0.05 / dt)
        
    def set_dt(self, dt):
        self.DT = dt
        self.dt = self.DT
        self.env.model.opt.timestep = dt / self.env.frame_skip
        self.max_episode_length = int(1000 * 0.05 / dt)

    def step(self,action):
        
        ob, reward, done, info = self.env.step(action)
        # print(reward)
        self.counter += 1
        if self.counter >= self.max_episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        # print(self.counter)
        return ob, reward , done, info

    def reset(self):
        self.counter = 0
        return self.env.reset()
    def seed(self, i):
        self.env.seed(i)

    def render(self, mode):
        return self.env.render(mode)

class CT_hopper:
    def __init__(self, dt=0.008) -> None:
        self.env = HopperEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        # self.dt = dt
        self.DT = dt
        self.dt = self.DT
        
        self.env.model.opt.timestep = dt / self.env.frame_skip
        self.counter = 0
        self.max_episode_length = int(1000 * 0.008 / dt)
        print(self.max_episode_length)
    def set_dt(self, dt):
        self.DT = dt
        self.dt = self.DT
        self.env.model.opt.timestep = dt / self.env.frame_skip
        self.max_episode_length = int(1000 * 0.008 / dt)

    def step(self,action):
        
        ob, reward, done, info = self.env.step(action)
        # print(reward)
        self.counter += 1
        if self.counter >= self.max_episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        # print(self.counter)
        return ob, reward , done, info

    def reset(self):
        self.counter = 0
        return self.env.reset()
    def seed(self, i):
        self.env.seed(i)

    def render(self, mode):
        return self.env.render(mode)



class CT_open_drawer:
    def __init__(self,dt = 0.04) -> None:
        
        self.env = gym.make('open_drawer-state-v0')#, render_mode='human')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.DT = 0.04
        self.dt = 0.04
        self.episode_length = 100
        self.counter = 0

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        self.DT = 0.04
        self.dt = 0.04
        return self.env.reset()

    def step(self,action,d=None):

        self.counter +=1 
        obs, reward, done, info = self.env.step(action)
        
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return obs, reward, done, info

class CT_close_drawer:
    def __init__(self,dt = 0.05) -> None:
        
        self.env = gym.make('close_drawer-state-v0')#, render_mode='human')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.DT = dt
        self.dt = dt
        self.episode_length = int(5 / dt) 
        self.counter = 0

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        # self.DT = 0.04
        self.dt = self.DT
        return self.env.reset()
    
    def render(self, mode):
        return self.env.render(mode)
    
    def step(self,action,d=None):

        self.counter +=1 
        action[-1] = 1
        obs, reward, done, info = self.env.step(action)
        reward = reward / self.DT
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return obs, reward, done, info


class CT_mountain_car(Continuous_MountainCarEnv):
    def __init__(self, dt=0.01):
        self.DT = dt
        self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        super().__init__()
        
        
        print(self.DT)

    def reset(self):
        self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        return super().reset()

    def step(self, action, d=None):
        if d is not None:
            self.dt = d
        else:
            self.dt = self.DT

        self.ep_time += self.dt
        self.counter += 1
        

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += self.dt / 0.01 * (force*self.power -0.0025 * np.cos(3*position))
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += self.dt / 0.01 * velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)

        reward = 0
        if done:
            reward = 100.0 * 0.01 / self.dt
        reward-= (action[0]**2) * 0.1

        self.state = np.array([position, velocity])
        s = self.state
        r = reward
        info = {}



        if self.ep_time >= 10:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return s,r,done,info

class CT_pendulum(PendulumEnv):
    def __init__(self, g=10, dt=0.05):
        super().__init__(g)
        self.ep_time = 0
        
        self.DT = dt
        
        self.dt = self.DT
        print(self.dt)

    def reset(self):
        self.dt = self.DT
        self.ep_time = 0
        return super().reset()

    def step(self, u, d=None):
        if d is not None:
            self.dt = d
        else:
            self.dt = self.DT
        self.ep_time += self.dt
        
        s,r,done,info = super().step(u)

        if self.ep_time >= (10-1e-3):
            done = True
            info['TimeLimit.truncated'] = True

        return s,r,done,info

        
class CT_pendulum_smooth(PendulumEnv):
    def __init__(self, g=10, dt=0.05):
        super().__init__(g)
        self.ep_time = 0
        
        self.DT = dt
        
        self.dt = self.DT
        print(self.dt)

    def reset(self):
        self.dt = self.DT
        self.ep_time = 0
        self.u_prev = None
        return super().reset()

    def step(self, u, d=None):
        if d is not None:
            self.dt = d
        else:
            self.dt = self.DT
        self.ep_time += self.dt
        
        s,r,done,info = super().step(u)
        if self.u_prev is not None:
            r -= 0.1 * np.abs(u-self.u_prev)[0] / self.dt
        self.u_prev = u
        if self.ep_time >= (10-1e-3):
            done = True
            info['TimeLimit.truncated'] = True

        return s,r,done,info

class CT_pendulum_sparse(PendulumEnv):
    def __init__(self, g=10):
        super().__init__(g)
        self.ep_time = 0
        print(self.dt)

    def reset(self):
        self.dt = 0.05
        self.ep_time = 0
        return super().reset()

    def step(self, u, d=None):
        if d is not None:
            self.dt = d
        else:
            self.dt = 0.05
        self.ep_time += self.dt
        th, thdot = self.state
        r_ = (((th+np.pi) % (2*np.pi)) - np.pi)**2
        
        if np.abs (((th+np.pi) % (2*np.pi)) - np.pi) < 0.1:
            # print('aaa')
            r_ += 1
        
        s,r,done,info = super().step(u)
        r_sparse = r_ + r
        if self.ep_time >= 10:
            done = True
            info['TimeLimit.truncated'] = True

        return s, r_sparse, done, info
        
class CT_sine:
    def __init__(self,dt = 0.01) -> None:
        
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.DT = 0.01
        self.dt = 0.01
        self.list = []
        self.episode_length = 100
        self.counter = 0
        

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        self.list = []
        self.DT = 0.01
        self.dt = 0.01
        self.F = 1 #np.random.random()/2+0.5
        self.A = 1#np.random.random()/2 + 0.5
        self.phi = np.random.random() * np.pi
        x = np.linspace(0,1,self.episode_length)
        self.traj = self.A * np.sin(2 * np.pi * x * self.F + self.phi)
        return np.array([self.phi, 0.])

    def step(self,action,d=None):

        self.counter +=1 
        self.list.append(action[0])
        reward = 0
        done = False
        if self.counter == self.episode_length:
            # print(np.array(self.list).shape)
            error = np.abs((np.array(self.list) - self.traj)**2).mean(axis=-1)#/(self.A**2)
            # print(error)
            # if error < 0.25:
            #     reward = 1/self.dt
            # plt.plot(self.list)
            # plt.savefig('sine.png')
                #print(np.array(self.list) - self.traj)
            # reward = -error/self.dt

            
        info = {}
        
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return np.array([self.phi, self.counter/ self.episode_length]), reward, done, info
    def seed(self,i):
        pass

class CT_sine_vel:
    def __init__(self,dt = 0.05) -> None:
        
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32)
        self.DT = 0.05
        self.dt = 0.05
        self.list = []
        self.episode_length = 20
        self.counter = 0
        

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        self.list = []
        self.DT = 0.05
        self.dt = 0.05
        self.F = 1 #np.random.random()/2+0.5
        self.A = 1#np.random.random()/2 + 0.5
        self.phi = (np.random.random() - 0.5) * 2 * np.pi
        x = np.linspace(0,1,20)
        self.traj = self.A * np.sin(2 * np.pi * x * self.F + self.phi)
        self.x = self.traj[0]
        return np.array([self.phi, 0., self.x])

    def step(self,action,d=None):

        self.counter +=1 
        self.x += 2*np.pi *self.dt * action[0]
        self.list.append(self.x)
        reward = 0
        done = False
        if self.counter == self.episode_length:
            # print(np.array(self.list).shape)
            error = np.abs((np.array(self.list) - self.traj)**2).mean(axis=-1)#/(self.A**2)
            # print(error)
            if error < 0.25:
                reward = 1/self.dt
                # plt.plot(self.list)
                # plt.savefig('sine.png')
                #print(np.array(self.list) - self.traj)
            # reward = -error/self.dt

            
        info = {}
        
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return np.array([self.phi, self.counter/ self.episode_length, self.x]), reward, done, info

class CT_sine_dense:
    def __init__(self,dt = 0.05) -> None:
        
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.DT = 0.05
        self.dt = 0.05
        self.list = []
        self.episode_length = 20
        self.counter = 0
        

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        self.list = []
        self.DT = 0.05
        self.dt = 0.05
        self.F = 1 #np.random.random()/2+0.5
        self.A = 1#np.random.random()/2 + 0.5
        self.phi = np.random.random() * np.pi
        x = np.linspace(0,1,20)
        self.traj = self.A * np.sin(2 * np.pi * x * self.F + self.phi)
        return np.array([self.phi, self.counter/ self.episode_length])

    def step(self,action,d=None):

        self.counter +=1 
        self.list.append(action[0])
        reward = 0
        done = False
        if self.counter == self.episode_length:
            # print(np.array(self.list).shape)
            error = np.abs((np.array(self.list) - self.traj)**2).mean(axis=-1)#/(self.A**2)
            # print(error)
            if error < 0.25:
                reward = 1/self.dt
                # plt.plot(self.list)
                # plt.savefig('sine.png')
                #print(np.array(self.list) - self.traj)
            # reward = -error/self.dt

        reward = -(action[0] - self.traj[self.counter-1])**2
        print(reward)
        info = {}
        
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return np.array([self.phi, self.counter/ self.episode_length]), reward, done, info

class Env_test:
    def __init__(self) -> None:
        # self.s = 2 + np.random.rand(1)
        self.dt = 0.1
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(1,), dtype=np.float32)

    def reset(self,):
        self.time = 0
        self.goal = np.array([1.])
        self.s = np.array([0.])
        # return np.concatenate((self.goal, self.s))
        return self.s

    def step(self, a, d=None):
        
        if d is not None:
            self.s += a * d
            self.time += d
        else:
            self.s += a * self.dt
            self.time += self.dt
        # print(self.t)
        # s = np.concatenate((self.goal, self.t))
        
        r = -((self.goal - self.s)**2).sum()
        done = False
        if self.time >= 5:
            done = True
        info = {}
        # print('r: ', r)
        # print(self.time, self.s, d, a * self.dt)
        return self.s ,r,done,info

class D2C:
    def __init__(self, discrete_env, low_level_funciton, rho, precise=False, eval_mode=False) -> None:
        self.env = discrete_env
        self.dt = discrete_env.dt  # discrete time-step
        self.rho = rho

        self.low_level_func = low_level_funciton
        self.precise = precise
        self.eval_mode = eval_mode
    def reset(self,):

        # test for pendulum
        # self.env.reset()
        # self.env.state[0] = np.pi/2
        # self.env.state[1] = 0.
        
        # theta, thetadot = self.env.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        ##

        return self.env.reset()

    def get_continuous_reward(self, r, start_t, dt):
        return r/self.rho * (np.exp(-self.rho * start_t) - np.exp(-self.rho*(start_t + dt)))
        # return dt * r * np.exp(-self.rho * start_t)

    def step(self, A, duration , option_duration = None):
        ### inputs: A is the action (high or low) to be repeated
        ###         duration is the duration of continuous time-step
        ###         Duration is the duration of option, Note: it can be none means there is no termination checking 
        ### outputs: R is the integral of e^(-rho t)x r(t) over period d
        ###          S is the state at the end of continuous time-step
        ###          Done is true if terminal happend 
        
        if option_duration is None:
            option_duration = duration
        
        R = 0
        Info = {'rewards':[],
                'durations': [],
                'frames': [],
                'actions': [],
                }
        done = False
        # duration = max(duration ,self.dt)
        
        integration_steps = int(duration / self.dt)
        if integration_steps == 0 and self.precise == False:
            integration_steps = 1
        for i in range(integration_steps):
            
            a = self.low_level_func(A, i * self.dt, option_duration)
            s, r, done, info = self.env.step(a)
            if self.eval_mode:
                f = self.env.render(mode='rgb_array')
                Info['frames'].append(f)
                Info['actions'].append(a)
            R += self.get_continuous_reward(r, i * self.dt, self.dt) 
            Info['rewards'].append(r)
            Info['durations'].append(self.dt)
            if done:
                Info['TimeLimit.truncated'] =  info['TimeLimit.truncated']
                return (s, R, done, Info)
        d = duration - integration_steps * self.dt
        
        if self.precise:
            if d > 0:      
                a = self.low_level_func(A, integration_steps * self.dt, option_duration)
                s, r, done, info = self.env.step(a, d)
                if self.eval_mode:
                    f = self.env.render(mode='rgb_array')
                    Info['frames'].append(f)
                    Info['actions'].append(a)
                R += self.get_continuous_reward(r, integration_steps * self.dt, d)
                Info['rewards'].append(r)
                Info['durations'].append(d)
            if done:
                Info['TimeLimit.truncated'] =  info['TimeLimit.truncated']
        return (s, R, done, Info)

class CT_stochastic_env:
    def __init__(self,dt = 0.01) -> None:
        
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.DT = dt
        self.dt = dt
        self.episode_length = int(1./dt)
        self.counter = 0
        self.change_goal_prob = 1./(1.+0.5/dt)
        

    def reset(self):
        # self.dt = self.DT
        self.counter = 0
        self.ep_time = 0
        
        # self.DT = 0.1
        self.dt = self.DT
        self.goal = 2 * np.random.random() - 1
        self.x = 0.0#0.2 * np.random.random() - 0.1
        self.state = np.array([self.x, self.goal])
        return self.state

    def step(self,action,d=None):

        self.counter +=1 
        reward = 0
        done = False
        

        self.x += action[0] * self.dt
        if np.random.random() > (1-self.change_goal_prob):
            print('changed', self.counter)
            self.goal = 2 * np.random.random() - 1
        self.state = np.array([self.x, self.goal])
        reward = - (self.x - self.goal) ** 2

        info = {}
        
        if self.counter == self.episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            if done:
                info['TimeLimit.truncated'] = False
        return self.state, reward, done, info
    def seed(self, i):
        pass

if __name__ == '__main__':
    env = CT_hopper_dm(dt = 0.01)
    
    s= env.reset()
    for i in range(1000):
        a = env.action_space.sample()
        s1,r1,done, info = env.step(a)
        # env.env.physics.render()
    
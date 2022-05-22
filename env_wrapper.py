import imp
import numpy as np
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control import Continuous_MountainCarEnv

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
        reward-= (action[0]**2) *0.1

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

        if self.ep_time >= 10:
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
    def __init__(self, discrete_env, low_level_funciton, rho, precise=False) -> None:
        self.env = discrete_env
        self.dt = discrete_env.dt  # discrete time-step
        self.rho = rho

        self.low_level_func = low_level_funciton
        self.precise = precise

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

    def step(self, A, duration ):
        ### inputs: A is the action (high or low) to be repeated
        ###         d is the duration of continuous time-step
        ### outputs: R is the integral of e^(-rho t)x r(t) over period d
        ###          S is the state at the end of continuous time-step
        ###          Done is true if terminal happend 
        R = 0
        Info = {'rewards':[],
                'durations': [],
                }
        done = False
        # duration = max(duration ,self.dt)
        
        integration_steps = int(duration / self.dt)
        for i in range(integration_steps):
            
            a = self.low_level_func(A, i * self.dt, duration)
            s, r, done, info = self.env.step(a)
            # self.env.render()
            R += self.get_continuous_reward(r, i * self.dt, self.dt) 
            Info['rewards'].append(r)
            Info['durations'].append(self.dt)
            if done:
                Info['TimeLimit.truncated'] =  info['TimeLimit.truncated']
                return (s, R, done, Info)
        d = duration - integration_steps * self.dt
        
        if self.precise:
            if d > 0:
                a = self.low_level_func(A, integration_steps * self.dt, duration)
                s, r, done, info = self.env.step(a, d)
                R += self.get_continuous_reward(r, integration_steps * self.dt, d)
                Info['rewards'].append(r)
                Info['durations'].append(d)
            if done:
                if info['TimeLimit.truncated']:
                    Info['TimeLimit.truncated'] = True
        return (s, R, done, Info)
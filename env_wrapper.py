import numpy as np
from gym import spaces

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
    def __init__(self, discrete_env, low_level_funciton, rho) -> None:
        self.env = discrete_env
        self.dt = discrete_env.dt  # discrete time-step
        self.rho = rho

        self.low_level_func = low_level_funciton

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
                'durations': []}
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
                return (s, R, done, Info)
        d = duration - integration_steps * self.dt
        if d > 0:
            a = self.low_level_func(A, integration_steps * self.dt, duration)
            s, r, done, info = self.env.step(a, d)
            R += self.get_continuous_reward(r, integration_steps * self.dt, d)
            Info['rewards'].append(r)
            Info['durations'].append(d)
        return (s, R, done, Info)
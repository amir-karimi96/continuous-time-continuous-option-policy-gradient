import numpy as np
from gym import spaces
class Env_test:
    def __init__(self) -> None:
        # self.s = 2 + np.random.rand(1)
        self.dt = 0.1
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(1,), dtype=np.float32)

    def reset(self,):
        self.counter = 0
        self.goal = np.array([1.])
        self.s = np.array([0.])
        # return np.concatenate((self.goal, self.s))
        return self.s

    def step(self, a, d=None):
        self.counter += 1
        if d is not None:
            self.s += a * d
        else:
            self.s += a * self.dt
        # print(self.t)
        # s = np.concatenate((self.goal, self.t))
        
        r = -((self.goal - self.s)**2).sum()
        done = False
        if self.counter == 50:
            done = True
        info = {}
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

    def get_continuous_reward(self, r, dt):
        return r * np.exp(- self.rho * dt) * self.dt

    def step(self, A, duration ):
        ### inputs: A is the action (high or low) to be repeated
        ###         d is the duration of continuous time-step
        ### outputs: R is the integral of e^(-rho t)x r(t) over period d
        ###          S is the state at the end of continuous time-step
        ###          Done is true if terminal happend 
        R = 0
        Info = {'rewards':[]}
        done = False
        duration = max(duration ,self.dt)
        
        actual_duration = int(duration / self.dt)
        for i in range(actual_duration):
            
            a = self.low_level_func(A, i * self.dt, duration)
            s, r, done, info = self.env.step(a)
            # self.env.render()
            R += self.get_continuous_reward(r, (i+1) * self.dt) 
            Info['rewards'].append(r)
            if done:
                return (s, R, done, Info)
        d = duration - actual_duration * self.dt
        if d > 0:
            a = self.low_level_func(A, actual_duration * self.dt, duration)
            s, r, done, info = self.env.step(a, d)
            R += self.get_continuous_reward(r, (actual_duration+1) * self.dt)
            Info['rewards'].append(r)
        return (s, R, done, Info)
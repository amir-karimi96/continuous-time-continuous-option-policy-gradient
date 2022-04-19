import numpy as np

class Env_test:
    def __init__(self) -> None:
        self.t = np.array([0., 1.])
    def reset(self,):
        self.counter = 0
        self.goal = np.array([-2., 2.])
        self.t = np.array([0., 1.])
        return np.concatenate((self.goal, self.t))

    def step(self, a):
        self.counter += 1
        
        self.t += a * 0.1
        # print(self.t)
        s = np.concatenate((self.goal, self.t))
        r = -((self.goal - self.t)**2).sum()
        done = False
        if self.counter == 3:
            done = True
        info = {}
        return s,r,done,info

class D2C:
    def __init__(self, discrete_env, low_level_funciton) -> None:
        self.env = discrete_env
        self.dt = discrete_env.dt  # discrete time-step
        
        self.low_level_func = low_level_funciton

    def reset(self,):
        return self.env.reset()
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
        for i in range( int(duration / self.dt)):
            
            a = self.low_level_func(A, i * self.dt, duration)
            s, r, done, info = self.env.step(a)
            # self.env.render()
            R += r * np.exp(-(i+1) * self.dt) * self.dt
            Info['rewards'].append(r)
            if done:
                break
            
        
        return (s, R, done, Info)
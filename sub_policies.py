import numpy as np
import matplotlib.pyplot as plt
import time

# definition of low level functions with input of action , current time, duration 

def rbf(centers, width):   
    
    b = lambda x: np.stack([np.exp(-(x - c_i) ** 2 / (2 * h_i)) for c_i, h_i in zip(centers, width)]).T  # eq 7
    return lambda x: b(x) / np.sum(b(x), axis=1, keepdims=True)  # eq 8

class sub_policy:
    def __init__(self, low_level_function_choice, low_level_action_dim, n_features= 5) -> None:
        self.low_level_action_dim = low_level_action_dim
        self.n_features = n_features
        self.low_level_function = self.__getattribute__(low_level_function_choice)
        # print(self.low_level_function(A = np.array([1,2,0.5,0,-1,1,-1,2,4,2]), t= 1, duration = 1))

    def repeat_action(self,A,t,duration):
        return A

    def linear_rbf(self, A, t, duration):
        
        n_features = self.n_features

        params = A.reshape(self.low_level_action_dim, n_features)
        
        h = (1. + 1 / n_features) / n_features 
        h = h ** 2 * 0.5

        bandwidths = np.repeat([h], n_features, axis=0)
        centers = np.linspace(0 - 0.5 / n_features, 1 + 0.5 / n_features, n_features)
        
        phi = rbf(centers, bandwidths)([t/duration])
        # print(phi.shape)
        # return np.matmul(phi, np.ones((5,1))) 
        # print(np.matmul(phi, params.T).shape)
        # print(np.matmul(phi, params.T).reshape(-1) , A, t, duration)
        return np.matmul(phi, params.T).reshape(-1)

def test(f):
    print(f(A = np.array([-1.1,-0.9,-1,-1,-1]), t= i, duration = 100))

if __name__ == '__main__':
    s= sub_policy('linear_rbf', 1)

    print(s.low_level_function(A = np.array([1,-1,2,4,2]), t= 1, duration = 1))
    y = []
    t0 = time.time()
    for i in range(100):
        r = s.low_level_function(A = np.array([-1.1,-0.9,-1,-1,-1]), t= i, duration = 100)
        y.append(r[0])

    print(time.time() - t0)
    plt.plot(y)
    plt.savefig('sub_policies_test.png')
    
from ast import parse
from math import exp
from typing_extensions import final
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape, var
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',default='0',type=str, help='exp file name')
parser.add_argument('--success_rate',default=0,type=int, help='bool ')
parser.add_argument('--window',default=100,type=int, help='window size ')
parser.add_argument('--experiments_directory', default='0', type=str, help = 'experiments dir')
args = parser.parse_args()
SR = args.success_rate
exp_name = args.exp_name
exps_dir = args.experiments_directory


with open('{}/{}/config_base.yaml'.format(exps_dir, exp_name),'r') as file:
    config_base = yaml.full_load(file)
    num_runs = config_base['experiment']['num_runs']
    num_params = config_base['experiment']['num_params']

fig = plt.figure()

ax = [fig.add_subplot(3, 1, 1), fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)]

D = [ [] for i in range(num_params)] # list for data
A = [ [] for i in range(num_params)] # list for actions

P = [ [] for i in range(num_params)] # for param config
configs = []

result = {  'exp_name': exp_name, 
            'config_base': config_base,
            'auc_perf': None, 
            'final_perfs': None, 
            'final_perf_stds': None, 
            'success_rates': None}



for p in range(num_params):
    with open('{}/{}/configs/config_{}.yaml'.format(exps_dir, exp_name,p),'r') as file:
        config = yaml.full_load(file)

    configs.append(config)
    print('% {}'.format(p/num_params))
    for i in range(num_runs):
        try:
            d = np.load('{}/{}/results/config_{}/data/{}.npy'.format(exps_dir, exp_name,p, i), allow_pickle=True)
        except:
            print('config_{} run {} not found'.format(p,i))
        else:
            d = d.item()
            D[int(d['config_ID'])].append(d['data'])
            if SR:
                A[int(d['config_ID'])].append(d['action_features'])
            P[int(d['config_ID'])]=d['config']

for i in range(num_params):
    lens = [j.shape[0] for j in D[i]]
    print(lens)
    #print(lens)
    min_len = min(lens)
    #len = min(len, 250000)
    print(min_len)
    D[i] = [j[:min_len] for j in D[i]] 
    D[i] = np.array(D[i])
    if SR:
        A[i] = np.array(A[i])
    

#print(D[1].shape)

# plot learning curves
final_perf = []
final_er = []
auc_perf = []
auc_er = []
for i,d in enumerate(D):
    #print(d.shape)
    # d = np.concatenate(list(d),axis=1)
    # print(d.shape)
    w = args.window
    R_mean = d.mean(axis = 0)
    
    R_std = d.std(axis = 0,ddof=1)/np.sqrt(d.shape[0])
    R_std_smoothed = np.convolve(R_std, np.ones(w), 'valid') / w
    R_smoothed = np.convolve(R_mean, np.ones(w), 'valid') / w
    final_perf.append(R_smoothed[-1])
    final_er.append(R_std_smoothed[-1])
    
    auc_perf.append(R_mean.sum())
    
    label = ''
    # for k in P[i]:
    #   label +=  str(k) + '_' +str(P[i][k]) + '_' 
    label = 'config_{}'.format(i)
    # if i in [4,6,7,15]:
    p = ax[0].plot(R_smoothed, label = label)
    ax[0].fill_between(range(R_smoothed.shape[0]), R_smoothed-R_std_smoothed, R_smoothed+R_std_smoothed, alpha = 0.3, color = p[0].get_color())

# store results

result['final_perfs'] = final_perf
result['final_perf_stds'] = final_er
result['auc_perf'] = auc_perf
# ax[0].legend()

final_perf = np.array(final_perf)
ax[0].set_xlabel('time-steps')
ax[0].set_ylabel('Avg Reward')

# ax[0].set_ylim([-1,0])
# ax[1].set_ylim([-1,0])

ax[1].errorbar(range(len(configs)), final_perf, final_er, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
#ax[0].set_yscale("log")
#ax[1].set_yscale("log")


# plot success rates
if SR:
    success_prop = np.zeros((num_params,))
    for i,d in enumerate(A):
       
        #,J = np.where( )
        #print(I)
        k=(np.abs(A[i].squeeze(-1)-0.9) < 0.025) +  (np.abs(A[i].squeeze(-1)+0.9) < 0.025)
        #I = np.where( k.sum(axis=1) )
        
        #configs[i]['param']['N_a']
        success_prop[i] = k.any(axis=1).mean()#len(I)/( num_runs)
        #success_prop[i] = final_perf[i]
    result['success_rates'] = success_prop
    ax[2].bar(range(num_params),success_prop)
    ax[2].set_ylim([0,1])

#plt.suptitle('N_actions = 10, N_modes = 4')
plt.savefig('{}/{}/{}.png'.format(exps_dir, exp_name,exp_name))
# np.save('/home/amirk96/projects/def-ashique/amirk96/CTCO/Experiment_results/data/{}.npy'.format(exp_name ), result)

def VS(performance_vector ,variable):
    best_ind_list = []
    for p in config_base['experiment']['params'][variable]:
        c_ind_list = []
        for i in range(num_params):
            if configs[i]['param'][variable] == p:
                c_ind_list.append(i)
        print(p, c_ind_list)
        best_ind_list.append(c_ind_list[np.argmax(performance_vector[c_ind_list])])
    return performance_vector[best_ind_list]
    



variables = []
for p in list(config_base['experiment']['params'].keys()):
    if len(config_base['experiment']['params'][p]) > 1:
        variables.append(p)

fig = plt.figure()

ax = [fig.add_subplot(1,len(variables), i+1) for i in range(len(variables))]


print(variables)
for i,v in enumerate(variables):
    y = VS(final_perf, v)
    ax[i].plot([p for p in config_base['experiment']['params'][v]] ,y,  linestyle='dashed', marker='o')
    ax[i].set_xlabel(v)
    ax[i].set_ylabel('best performance')
    # ax[i].set_xscale("log")
plt.tight_layout()
plt.savefig('{}/{}/{}.pdf'.format(exps_dir, exp_name,exp_name+'_VA'))
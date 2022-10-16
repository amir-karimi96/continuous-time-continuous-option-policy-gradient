from ast import parse
from math import exp
from typing_extensions import final
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape, var
import yaml
import argparse
from scipy.stats import binned_statistic


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

ax = [fig.add_subplot(1, 1, 1)]#, fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)]

D = [ [] for i in range(num_params)] # list for data
D_time = [ [] for i in range(num_params)] # list for data_time

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
            if d['data'].shape[0] == 1:
                print(int(d['config_ID']), i)
            # D[int(d['config_ID'])].append(d['data'])
            D[int(d['config_ID'])].append(d['returns_discounted'])
            
            D_time[int(d['config_ID'])].append(d['data_wall_time'])
            
            if SR:
                A[int(d['config_ID'])].append(d['action_features'])
            P[int(d['config_ID'])]=d['config']
        


for i in range(num_params):
    lens = [j.shape[0] for j in D[i]]
    times = [T[-1] for T in D_time[i]]
    # print(i,lens)
    #print(lens)
    min_len = min(lens)
    min_time = min(times)
    #len = min(len, 250000)
    print(i,np.argmin(lens), lens)
    D[i] = [j[:np.where( T <= min_time)[0][-1]] for j,T in zip(D[i], D_time[i])] 
    # print(np.array(D_time[i]))
    # print(D_time[i][0])
    # print('time_last: ', np.where( D_time[i][0] <= min_time)[0][-1])
    D_time[i] = [T[:np.where( T <= min_time)[0][-1]] for T in D_time[i]] 
    # print(D_time[i])
    # exit()
    # D[i] = np.array(D[i])
    # D_time[i] = np.array(D[i])
    
    

#print(D[1].shape)

# plot learning curves
final_perf = []
final_er = []
auc_perf = []
auc_er = []

Retruns_vs_time = []
Retruns_error_vs_time = []
Times = []

R_mean_lsit = []
R_std_list = []
MAX_TIME = 108000
for i,d in enumerate(D):
    #print(d.shape)
    # d = np.concatenate(list(d),axis=1)
    # print(d.shape)
    runs_list = []
    T_min = 100
    for j,T in zip(D[i], D_time[i]):
        mean_stat = binned_statistic(T, j, 
                                statistic='mean', 
                                bins=MAX_TIME//60, 
                                range=(0, MAX_TIME))
        T_min = min(T_min, T[-1])
        #print(i,len(j), mean_stat.statistic)
        runs_list.append(mean_stat.statistic)
    runs_list = np.array(runs_list)
    print(i,runs_list[:,100])
    w = args.window
    R_mean = runs_list.mean(axis = 0)
    R_mean_lsit.append(R_mean)
    ###
    R_std = 1.96 * runs_list.std(axis = 0,ddof=1)/np.sqrt(runs_list.shape[0])
    R_std_list.append(R_std)
    R_std_smoothed = np.convolve(R_std, np.ones(w), 'valid') / w
    R_smoothed = np.convolve(R_mean, np.ones(w), 'valid') / w
    N_ind = np.isnan(R_smoothed).argmax()
    N_ind = min(400, N_ind)
    final_perf.append(R_smoothed[N_ind-1])
    final_er.append(R_std_smoothed[N_ind-1])
    
    auc_perf.append(R_mean.sum())
    Retruns_vs_time.append(R_mean)
    Retruns_error_vs_time.append(R_std)
    # Times.append()
    label = ''
    # for k in P[i]:
    #   label +=  str(k) + '_' +str(P[i][k]) + '_' 
    # print(configs[i]['param']['env_dt'])
    label = 'freq={}'.format(1/configs[i]['param']['env_dt'])
    # label = 'z_dim={}'.format(configs[i]['param']['z_dim'])
    # if i in [4,6,7,15]:
    p = ax[0].plot(R_mean, label = label)
    ax[0].fill_between(range(R_mean.shape[0]), R_mean-R_std, R_mean+R_std, alpha = 0.3, color = p[0].get_color())
    

# store results

result['final_perfs'] = final_perf
result['final_perf_stds'] = final_er
result['auc_perf'] = auc_perf
result['Returns_vs_time'] = Retruns_vs_time
result['Returns_error_vs_time'] = Retruns_error_vs_time

result['configs'] = configs
ax[0].legend()

final_perf = np.array(final_perf)
ax[0].set_xlabel('wall time (min)')
ax[0].set_ylabel('Avg Return')

# ax[0].set_ylim([-1,0])
# ax[1].set_ylim([-1,0])

# ax[1].errorbar(range(len(configs)), final_perf, final_er, fmt='o', color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0)
#ax[0].set_yscale("log")
#ax[1].set_yscale("log")


#plt.suptitle('N_actions = 10, N_modes = 4')
plt.savefig('{}/{}/{}.png'.format(exps_dir, exp_name,exp_name))

plt.savefig('{}/{}/{}.pdf'.format(exps_dir, exp_name,exp_name))
np.save('{}/{}/{}.npy'.format(exps_dir, exp_name,exp_name), result)

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
    

def plot_():
    penalty_list = config_base['experiment']['params']['Duration_penalty_const']
    z_dim_list = config_base['experiment']['params']['z_dim']
    env_dt_list = config_base['experiment']['params']['env_dt']
    c = np.ndarray((len(penalty_list), len(z_dim_list)),dtype=list)
    fig, ax = plt.subplots(len(penalty_list), len(z_dim_list),sharey=True, sharex=True)
    print(ax.shape)
    for i,penalty in enumerate(penalty_list):
        for j,z in enumerate(z_dim_list):
            c[i,j] = []
            # print(p, z)
            for k in range(num_params):
                if configs[k]['param']['Duration_penalty_const'] == penalty and configs[k]['param']['z_dim'] == z:
                    c[i,j].append(k)
            print(c[i,j])
            for ind in c[i,j]:
                # ax[i,j].plot([1,2,3])
                print(ind)
                p = ax[i,j].plot(R_mean_lsit[ind],label='freq = {}'.format(1/configs[ind]['param']['env_dt']))
                ax[i,j].fill_between(range(R_mean_lsit[ind].shape[0]), R_mean_lsit[ind]-R_std_list[ind], R_mean_lsit[ind]+R_std_list[ind], alpha = 0.3, color = p[0].get_color())
                ax[i,j].set_title('duration_penalty = {}, z_dim = {}'.format(penalty, z),fontsize=7)
    ax[-1,-1].legend()
    plt.tight_layout()
    plt.savefig('{}/{}/{}_spec.png'.format(exps_dir, exp_name,exp_name))
    plt.savefig('{}/{}/{}_spec.pdf'.format(exps_dir, exp_name,exp_name))

    
plot_()
exit()
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
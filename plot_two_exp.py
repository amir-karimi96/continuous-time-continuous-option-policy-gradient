import numpy as np
import matplotlib.pyplot as plt

exps_dir = '/home/amirk96/scratch/CTCO_Experiments'
exp_name1 = 'repeat_action_mountain_car'
exp_name2 = 'SAC_async_mountain_car' 
d1 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name1,exp_name1), allow_pickle=True)
d2 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name2,exp_name2), allow_pickle=True)
d1 = d1.item()
d2 = d2.item()

# print(d2['env_dt'])
# print(d2)
freqs1 = 1/np.array(d1['config_base']['experiment']['params']['env_dt'])
freqs2 = 1/np.array(d2['config_base']['experiment']['params']['env_dt'])

# plt.plot(freqs1, d1['final_perfs'])
# plt.plot(freqs2, d2['final_perfs'])
plt.errorbar(freqs1, d1['final_perfs'], d1['final_perf_stds'], fmt='o', color='blue',
             ecolor='lightgray', elinewidth=3, capsize=0,label='CTCO')
plt.errorbar(freqs2, d2['final_perfs'], d2['final_perf_stds'], fmt='o', color='red',
             ecolor='lightgray', elinewidth=3, capsize=0,label='SAC_async')
plt.ylabel('Average Return')
plt.xscale("log")
plt.xlabel('Frequency')
plt.legend()
plt.savefig('r.png')
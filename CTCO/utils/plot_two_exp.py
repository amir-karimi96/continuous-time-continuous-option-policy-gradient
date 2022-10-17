import matplotlib
import numpy as np
import matplotlib.pyplot as plt

exps_dir = '/home/amirk96/scratch/CTCO_Experiments'
exp_name1 = 'repeat_action_mountain_car_sync'
exp_name2 = 'SAC_sync_mountain_car'
exp_name3 = 'FiGAR_sync_mountain_car'

d1 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name1,exp_name1), allow_pickle=True)
d2 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name2,exp_name2), allow_pickle=True)
d3 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name3,exp_name3), allow_pickle=True)

d1 = d1.item()
d2 = d2.item()
d3 = d3.item()
colors = [c for c in matplotlib.colors.TABLEAU_COLORS]
# print(d2['env_dt'])
# print(d2)
freqs1 = 1/np.array(d1['config_base']['experiment']['params']['env_dt'])
freqs2 = 1/np.array(d2['config_base']['experiment']['params']['env_dt'])
freqs3 = 1/np.array(d3['config_base']['experiment']['params']['env_dt'])
# plt.plot(freqs1, d1['final_perfs'])
# plt.plot(freqs2, d2['final_perfs'])

plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.errorbar(freqs1, d1['final_perfs'], d1['final_perf_stds'],label='CTCO')
plt.errorbar(freqs2, d2['final_perfs'], d2['final_perf_stds'],label='SAC_async')
plt.errorbar(freqs3, d3['final_perfs'], d3['final_perf_stds'],label='FiGAR')
# plt.errorbar(freqs1, d1['final_perfs'], d1['final_perf_stds'], fmt='s', color=colors[3],
#              ecolor='lightgray', elinewidth=3, capsize=0,label='CTCO',alpha=0.8,mec='black')
# plt.errorbar(freqs2, d2['final_perfs'], d2['final_perf_stds'], fmt='s', color=colors[1],
#              ecolor='lightgray', elinewidth=3, capsize=0,label='SAC_async',alpha=0.8,mec='black')
# plt.errorbar(freqs3, d3['final_perfs'], d3['final_perf_stds'], fmt='s', color=colors[2],
#              ecolor='lightgray', elinewidth=3, capsize=0,label='FiGAR',alpha=0.8,mec='black')

plt.ylabel('Average discounted Return')
plt.xscale("log")
plt.xlabel('Frequency (Hz) ')
plt.legend()
plt.xticks(np.arange(10,1000,50))
plt.xlim([10, 1000])
plt.tick_params(axis='x', which='minor')
plt.title('Final performance of algorithms on mountain car vs frequency')
plt.savefig('r.pdf')
plt.savefig('r.png')
import tikzplotlib
tikzplotlib.save("mytikz.tex")
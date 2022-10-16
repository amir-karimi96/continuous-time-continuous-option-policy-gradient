import matplotlib
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()

ax = [fig.add_subplot(1, 1, 1)]#, fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)]


exps_dir = '/home/amirk96/scratch/CTCO_Experiments'
exp_name1 = 'test_mountain_car'
exp_name2 = 'FiGAR_mountain_car_2'
# exp_name3 = 'FiGAR_sine'

d1 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name1,exp_name1), allow_pickle=True)
d2 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name2,exp_name2), allow_pickle=True)
# d3 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name3,exp_name3), allow_pickle=True)

d1 = d1.item()
d2 = d2.item()
# d3 = d3.item()
colors = [c for c in matplotlib.colors.TABLEAU_COLORS]
# print(d2['env_dt'])
# print(d2)
Returns_1 = d1['Returns_vs_time']
# print(Returns_1)
Returns_errors_1 = d1['Returns_error_vs_time']
Returns_errors_2 = d2['Returns_error_vs_time']

configs_1 = d1['configs']
Returns_2 = d2['Returns_vs_time']
configs_2 = d2['configs']

for i, (R_mean, R_std) in enumerate(zip(Returns_1, Returns_errors_1)):
    if i in [11,10,9] or 1:
        label = 'z_dim={}'.format(configs_1[i]['param']['z_dim'])
        # if i in [4,6,7,15]:
        p = ax[0].plot(R_mean, label = label)
        ax[0].fill_between(range(R_mean.shape[0]), R_mean-R_std, R_mean+R_std, alpha = 0.3, color = p[0].get_color())

for i, (R_mean, R_std) in enumerate(zip(Returns_2, Returns_errors_2)):
    label = 'FiGAR_C'#.format(configs_2[i]['param']['D_max'])
    # if i in [4,6,7,15]:
    p = ax[0].plot(R_mean, label = label)
    ax[0].fill_between(range(R_mean.shape[0]), R_mean-R_std, R_mean+R_std, alpha = 0.3, color = p[0].get_color())

# plt.errorbar(freqs1, d1['final_perfs'], d1['final_perf_stds'], fmt='s', color=colors[3],
#              ecolor='lightgray', elinewidth=3, capsize=0,label='CTCO',alpha=0.8,mec='black')
# plt.errorbar(freqs2, d2['final_perfs'], d2['final_perf_stds'], fmt='s', color=colors[1],
#              ecolor='lightgray', elinewidth=3, capsize=0,label='SAC_async',alpha=0.8,mec='black')
# plt.errorbar(freqs3, d3['final_perfs'], d3['final_perf_stds'], fmt='s', color=colors[2],
#              ecolor='lightgray', elinewidth=3, capsize=0,label='FiGAR',alpha=0.8,mec='black')

plt.ylabel('Average discounted Return')
# plt.xscale("log")
plt.xlabel('Time (min) ')
plt.legend()
# plt.xticks(np.arange(10,1000,50))
# plt.xlim([10, 1000])
# plt.tick_params(axis='x', which='minor')
plt.title('Final performance of algorithms on mountain_car')
plt.savefig('r3.pdf')
plt.savefig('r3.png')
import tikzplotlib
tikzplotlib.save("mytikz.tex")
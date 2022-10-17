import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(1, 3)
fig.suptitle('Final performance of algorithms vs frequency')
csfont = {'fontname':'Times New Roman'}
# plt.title('title',**csfont)

exps_dir = '/home/amirk96/scratch/CTCO_Experiments'
tasks = ['Mountain_car', 'Half_cheetah', 'Ball_in_cup']
exp_name1s = ['f_CTCO_mountain_car','test_half_cheetah_25sec' ,'extend_CTCO_ball_in_cup']
config_1_masks = [None,[13,15,17], None]
exp_name2s = ['f_SAC_mountain_car','f_SAC_half_cheetah', 'SAC_ball_in_cup']
exp_name3s = ['f_FiGAR_mountain_car','FiGAR_half_cheetah', 'FiGAR_ball_in_cup']
exp_name4s = ['ff_DAC_mountain_car','ff_DAC_half_cheetah', 'ff_DAC_ball_in_cup']
i = 0
for exp_name1,config_1_mask, exp_name2, exp_name3, exp_name4 in zip(exp_name1s,config_1_masks, exp_name2s, exp_name3s, exp_name4s):

    d1 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name1,exp_name1), allow_pickle=True)
    d2 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name2,exp_name2), allow_pickle=True)
    d3 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name3,exp_name3), allow_pickle=True)
    d4 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name4,exp_name4), allow_pickle=True)

    d1 = d1.item()
    d2 = d2.item()
    d3 = d3.item()
    d4 = d4.item()

    colors = [c for c in matplotlib.colors.TABLEAU_COLORS]
    # print(d2['env_dt'])
    # print(d2)
    freqs1 = 1/np.array(d1['config_base']['experiment']['params']['env_dt'])
    freqs2 = 1/np.array(d2['config_base']['experiment']['params']['env_dt'])
    freqs3 = 1/np.array(d3['config_base']['experiment']['params']['env_dt'])
    freqs4 = 1/np.array(d4['config_base']['experiment']['params']['env_dt'])
    p1 = freqs1.argsort()
    p2 = freqs2.argsort()
    

    # plt.plot(freqs1, d1['final_perfs'])
    # plt.plot(freqs2, d2['final_perfs'])

    ax[i].grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    if len(d1['final_perfs']) > len(freqs1):
        ax[i].errorbar(freqs1[p1], np.array(d1['final_perfs'])[config_1_mask][p1], np.array(d1['final_perf_stds'])[config_1_mask][p1],label='CTCO', fmt='-o')
    else: 
        ax[i].errorbar(freqs1[p1], np.array(d1['final_perfs'])[p1], np.array(d1['final_perf_stds'])[p1],label='CTCO', fmt='-o')

    ax[i].errorbar(freqs2[p2], np.array(d2['final_perfs'])[p2], np.array(d2['final_perf_stds'])[p2],label='SAC', fmt='-s')
    ax[i].errorbar(freqs3, d3['final_perfs'], d3['final_perf_stds'],label='FiGAR', fmt='-^')
    ax[i].errorbar(freqs4, np.array(d4['final_perfs'])*d4['config_base']['experiment']['params']['env_default_dt'], np.array(d4['final_perf_stds'])*d4['config_base']['experiment']['params']['env_default_dt'],label='DAC', fmt='-v')
    # ax[i].errorbar(freqs4, np.array(d4['final_perfs'])*d4['config_base']['experiment']['params']['env_default_dt']* 0.05 / d4['config_base']['experiment']['params']['env_dt'], np.array(d4['final_perf_stds'])*d4['config_base']['experiment']['params']['env_default_dt'] * 0.05 / d4['config_base']['experiment']['params']['env_dt'],label='DAC', fmt='-v')

    # plt.errorbar(freqs1, d1['final_perfs'], d1['final_perf_stds'], fmt='s', color=colors[3],
    #              ecolor='lightgray', elinewidth=3, capsize=0,label='CTCO',alpha=0.8,mec='black')
    # plt.errorbar(freqs2, d2['final_perfs'], d2['final_perf_stds'], fmt='s', color=colors[1],
    #              ecolor='lightgray', elinewidth=3, capsize=0,label='SAC_async',alpha=0.8,mec='black')
    # plt.errorbar(freqs3, d3['final_perfs'], d3['final_perf_stds'], fmt='s', color=colors[2],
    #              ecolor='lightgray', elinewidth=3, capsize=0,label='FiGAR',alpha=0.8,mec='black')

    ax[i].set_ylabel('Average discounted Return')
    ax[i].set_xscale("log")
    ax[i].set_xlabel('Frequency (Hz) ', **csfont)
    
    ax[i].set_xticks(np.arange(10,1000,50))
    ax[i].set_xlim([10, 1000])
    ax[i].tick_params(axis='x', which='minor')
    ax[i].set_title(tasks[i])
    i+=1
ax[-1].legend()
fig.set_size_inches(8.5, 4.5)
plt.tight_layout()
plt.savefig('tasks.pdf')
plt.savefig('tasks.png')
import tikzplotlib
tikzplotlib.save("mytikz_tasks.tex")
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
df = px.data.iris()
# fig_ = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
#                 "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
#                 "petal_width": "Petal Width", "petal_length": "Petal Length", },
#                              color_continuous_scale=px.colors.diverging.Tealrose,
#                              color_continuous_midpoint=2)
# print(df)
# fig_.write_image("sesnse.png")
fig, ax = plt.subplots(1, 3)
fig.suptitle('Final performance of algorithms vs frequency')


exps_dir = '/home/amirk96/scratch/CTCO_Experiments'
tasks = ['Mountain_car', 'Half_cheetah', 'Ball_in_cup']
exp_name1s = ['test_pendulum']#,'test_half_cheetah_25sec' ,'extend_CTCO_ball_in_cup']
config_1_masks = range(27)#[None,[13,15,17], None]

i = 0
for exp_name1,config_1_mask in zip(exp_name1s,config_1_masks):

    d1 = np.load('{}/{}/{}.npy'.format(exps_dir, exp_name1,exp_name1), allow_pickle=True)
    
    d1 = d1.item()
    # print(d1)
    colors = [c for c in matplotlib.colors.TABLEAU_COLORS]
    # print(d2['env_dt'])
    # print(d2)
    freqs1 = 1/np.array([d['param']['env_dt'] for d in d1['configs']])
    print(freqs1)
    rbfs = [d['param']['z_dim'] for d in d1['configs']]#d1['config_base']['experiment']['params']['z_dim']
    penalties = [d['param']['Duration_penalty_const'] for d in d1['configs']]#d1['config_base']['experiment']['params']['Duration_penalty_const']
    p1 = freqs1.argsort()
    print(len(freqs1),len(rbfs), len(penalties) )
    data = {'Frequency': freqs1,
            'N_RBF': rbfs,
            'penalty': penalties,
            'final_performance': d1['final_perfs']}
    fig_ = px.parallel_coordinates(data, color="final_performance", labels={
                "Frequency": "Frequency", "N_RBF": "N_RBF",
                "penalty": "penalty", "final_performance": "final_performance", },
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2,range_color=[-8,-5],)
    print(px.colors.diverging.Tealrose)
    fig_.write_image("sesnse.png")
    # plt.plot(freqs1, d1['final_perfs'])
    # plt.plot(freqs2, d2['final_perfs'])

    ax[i].grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    if len(d1['final_perfs']) > len(freqs1):
        ax[i].errorbar(freqs1[p1], np.array(d1['final_perfs'])[config_1_mask][p1], np.array(d1['final_perf_stds'])[config_1_mask][p1],label='CTCO', fmt='-o')
    else: 
        ax[i].errorbar(freqs1[p1], np.array(d1['final_perfs'])[p1], np.array(d1['final_perf_stds'])[p1],label='CTCO', fmt='-o')

    
    # plt.errorbar(freqs1, d1['final_perfs'], d1['final_perf_stds'], fmt='s', color=colors[3],
    #              ecolor='lightgray', elinewidth=3, capsize=0,label='CTCO',alpha=0.8,mec='black')
    # plt.errorbar(freqs2, d2['final_perfs'], d2['final_perf_stds'], fmt='s', color=colors[1],
    #              ecolor='lightgray', elinewidth=3, capsize=0,label='SAC_async',alpha=0.8,mec='black')
    # plt.errorbar(freqs3, d3['final_perfs'], d3['final_perf_stds'], fmt='s', color=colors[2],
    #              ecolor='lightgray', elinewidth=3, capsize=0,label='FiGAR',alpha=0.8,mec='black')

    ax[i].set_ylabel('Average discounted Return')
    ax[i].set_xscale("log")
    ax[i].set_xlabel('Frequency (Hz) ')
    
    ax[i].set_xticks(np.arange(10,1000,50))
    ax[i].set_xlim([10, 1000])
    ax[i].tick_params(axis='x', which='minor')
    ax[i].set_title(tasks[i])
    i+=1
ax[-1].legend()
fig.set_size_inches(8.5, 4.5)
plt.tight_layout()
# plt.savefig('tasks.pdf')
# plt.savefig('tasks.png')
# import tikzplotlib
# tikzplotlib.save("mytikz_tasks.tex")
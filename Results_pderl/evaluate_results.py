import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 300
# plt.rcParams['figure.figsize'] = [6, 5]


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
color_ddpg = colors[2]
color_erl = colors[0]

savefig = True

def plot_games(ddpg_score, ddpg_std, erl_score, erl_std, games_ddpg):
    fig1,ax = plt.subplots()
    fig1.canvas.manager.set_window_title("Reward versus frames")
    # ax.set_title("Average Reward - LunarLander", pad=20)
    ax.plot(games_ddpg, ddpg_score, label = 'DDPG', color = color_ddpg)
    ax.fill_between(games_ddpg, ddpg_score - ddpg_std, ddpg_score+ ddpg_std, color=color_ddpg,alpha=0.4)

    ax.plot(games_ddpg, erl_score, label = 'PD-ERL', color=color_erl)
    ax.fill_between(games_ddpg, erl_score - erl_std, erl_score+ erl_std, color=color_erl, alpha=0.4)
    ax.set_ylabel("Reward [-]")
    ax.set_xlabel(r"Games [-]")
    ax.legend(loc = 'lower right')
    fig1.tight_layout()

    if savefig:
        fig1.savefig('Results_pderl/Plots/reward_games.png')

    return fig1, ax


def plot_frames(ddpg_score, ddpg_std, erl_score, erl_std, frames_ddpg):
    f2e = 10**4   # frames to epochs contraction for better plotting
    fig2,ax = plt.subplots()
    fig2.canvas.manager.set_window_title("Reward versus games")
    # ax.set_title("Average Reward - LunarLander", pad=20)
    ax.plot(frames_ddpg//f2e, ddpg_score, label = 'DDPG', color=color_ddpg)
    ax.fill_between(frames_ddpg//f2e, ddpg_score - ddpg_std, ddpg_score + ddpg_std, color=color_ddpg,alpha=0.4)

    ax.plot(frames_ddpg//f2e, erl_score, label = 'PD-ERL', color=color_erl)
    ax.fill_between(frames_ddpg//f2e, erl_score - erl_std, erl_score + erl_std, color=color_erl, alpha=0.4)
    ax.set_ylabel("Reward [-]")
    ax.set_xlabel(r"Epochs [$10^4$ frames]")
    ax.legend(loc = 'lower right')
    fig2.tight_layout()

    if savefig:
        fig2.savefig('Results_pderl/Plots/reward_frames.png')

    return fig2, ax

def plot_fault_tolerancy():
    erl_r = 251.25; erl_std = 26.31
    ddpg_r = 140; ddpg_std = 110.31
    qd_r = 200; qd_std = 70.4
    # borken engine fault
    erl_r_faulty = 207.22; erl_std_faulty = 0.8* 104.17
    ddpg_r_faulty = 5.41; ddpg_std_faulty = 35.69
    qd_r_faulty = 44.25; qd_std_faulty = 150.30

    labels = ('PD-ERL', 'DDPG', 'QD')
    nominal_r= [erl_r, ddpg_r,qd_r]; nominal_std = [erl_std,ddpg_std, qd_std]
    faulty_r = [erl_r_faulty, ddpg_r_faulty,qd_r_faulty]; faulty_std = [erl_std_faulty,ddpg_std_faulty, qd_std_faulty]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, nominal_r, width, yerr = nominal_std, label='Nominal',  capsize=6, ecolor= (0,0,0,0.7))
    rects2 = ax.bar(x + width/2, faulty_r, width, yerr= faulty_std, label='Broken Engine',   capsize=6, ecolor = (0,0,0,0.7))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.bar_label(rects1, labels=[f'{e:.1f}' for e in nominal_std],
             padding=2, fontsize=14)
    ax.bar_label(rects2, labels=[f'{e:.1f}' for e in faulty_std],
             padding=2, fontsize=14)

    ax.set_ylabel('Rewards [-]')
    # ax.set_title('')
    ax.set_xticks(x, labels)
    ax.legend(loc = 'lower left')

    fig.tight_layout()
    if savefig:
        fig.savefig('Results_pderl/Plots/bar_chart.png')


#-----------------------------------------------------------------------------

# Load data
logs = 'pderl/logs_s1_e3_buffer5e04'
logs_ddpg = 'pderl/logs_ddpg'

# ddpg
ddpg_score = np.genfromtxt(logs_ddpg+ '/ddpg_reward_frames.csv', skip_header= 1, delimiter=',')
ddpg_std = np.genfromtxt(logs_ddpg + '/ddpg_std_games.csv', skip_header= 1, delimiter=',')

# erl
erl_score = np.genfromtxt(logs + '/erl_frames.csv', skip_header= 1, delimiter=',')
erl_std   = np.genfromtxt(logs + '/erl_std_games.csv', skip_header= 1, delimiter=',')


# Retrieve x-axis arrays:
frames_ddpg = ddpg_score[:,0]; frames_erl = erl_score[:,0]
games_erl = erl_std[:,0]; games_ddpg = ddpg_std[:,0]
ddpg_score = ddpg_score[:,1]; ddpg_std= ddpg_std[:,1]; 
print(f'Recorded frames: erl-{len(frames_erl)}, ddpg-{len(frames_ddpg)}')

# # Slicing:
extra = 0.05
assert len(games_ddpg) == len(frames_ddpg)
if frames_ddpg[-1] > frames_erl[-1]:
    max_f = np.argwhere(frames_ddpg > (1+extra) * frames_erl[-1])[0][0]
    frames_ddpg = frames_ddpg[:max_f]; games_ddpg = games_ddpg[:max_f]
    ddpg_score = ddpg_score[:max_f]; ddpg_std = ddpg_std[:max_f]
else:
    max_f = np.argwhere(frames_erl > (1+extra) * frames_ddpg[-1])[0][0]
    frames_erl = frames_erl[:max_f]; games_erl = games_erl[:max_f]
    erl_score = erl_score[:max_f]; erl_std = erl_std[:max_f]


# Resample:
erl_score = np.interp(frames_ddpg, frames_erl, erl_score[:,1])
erl_std   = np.interp(frames_ddpg, frames_erl, erl_std[:,1])


# Plotting:
do_plot = True
if do_plot:
    plot_games(ddpg_score, ddpg_std, erl_score, erl_std, games_ddpg)
    # plot_frames(ddpg_score, ddpg_std, erl_score, erl_std, frames_ddpg)
    plot_fault_tolerancy()
    plt.show()


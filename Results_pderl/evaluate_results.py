import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
plt.rcParams.update({'font.size': 12})

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

def plot_games(ddpg_score, ddpg_std, erl_score, erl_std, games_ddpg):
    fig1,ax = plt.subplots()
    fig1.canvas.manager.set_window_title("Reward versus frames")
    # ax.set_title("Average Reward - LunarLander", pad=20)
    ax.plot(games_ddpg, ddpg_score, label = 'DDPG', color = colors[2])
    ax.fill_between(games_ddpg, ddpg_score - ddpg_std, ddpg_score+ ddpg_std, color=colors[2],alpha=0.4)

    ax.plot(games_ddpg, erl_score, label = 'PD-ERL', color=colors[0])
    ax.fill_between(games_ddpg, erl_score - erl_std, erl_score+ erl_std, color=colors[0], alpha=0.4)
    ax.set_ylabel("Reward [-]")
    ax.set_xlabel(r"Games [-]")
    ax.legend(loc = 'lower right')
    fig1.tight_layout()
    fig1.savefig('Results_pderl/Plots/reward_games.png')

    return fig1, ax


def plot_frames(ddpg_score, ddpg_std, erl_score, erl_std, frames_ddpg):
    f2e = 10**4   # frames to epochs contraction for better plotting
    fig2,ax = plt.subplots()
    fig2.canvas.manager.set_window_title("Reward versus games")
    # ax.set_title("Average Reward - LunarLander", pad=20)
    ax.plot(frames_ddpg//f2e, ddpg_score, label = 'DDPG', color=colors[2])
    ax.fill_between(frames_ddpg//f2e, ddpg_score - ddpg_std, ddpg_score + ddpg_std, color=colors[2],alpha=0.4)

    ax.plot(frames_ddpg//f2e, erl_score, label = 'PD-ERL', color=colors[0])
    ax.fill_between(frames_ddpg//f2e, erl_score - erl_std, erl_score + erl_std, color=colors[0], alpha=0.4)
    ax.set_ylabel("Reward [-]")
    ax.set_xlabel(r"Epochs [$10^4$ frames]")
    ax.legend(loc = 'lower right')
    fig2.tight_layout()
    fig2.savefig('Results_pderl/Plots/reward_frames.png')

    return fig2, ax


#-----------------------------------------------------------------------------

# Load data
logs = 'pderl/logs_s1_e3_buffer5e04'
logs_ddpg = 'pderl/logs_ddpg'

ddpg_score = np.genfromtxt(logs_ddpg+ '/ddpg_reward_frames.csv', skip_header= 1, delimiter=',')
ddpg_std = np.genfromtxt(logs_ddpg + '/ddpg_std_games.csv', skip_header= 1, delimiter=',')

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
    plot_frames(ddpg_score, ddpg_std, erl_score, erl_std, frames_ddpg)

    plt.show()
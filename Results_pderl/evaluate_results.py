import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot') 
plt.rcParams.update({'font.size': 12})


# Load data
logs = 'pderl/logs_first_good'
logs_ddpg = 'pderl/logs_ddpg'

ddpg_score = np.genfromtxt(logs_ddpg+ '/ddpg_reward_frames.csv', skip_header= 1, delimiter=',')
ddpg_std = np.genfromtxt(logs_ddpg + '/ddpg_std_games.csv', skip_header= 1, delimiter=',')

erl_score = np.genfromtxt(logs + '/frame_erl_frames.csv', skip_header= 1, delimiter=',')
erl_std   = np.genfromtxt(logs + '/erl_std_games.csv', skip_header= 1, delimiter=',')


# Retrieve axis arrays:
frames_ddpg = ddpg_score[:,0]; frames_erl = erl_score[:,0]
games_erl = erl_std[:,0]; games_ddpg = ddpg_std[:,0]

# Slicing:
max_f = min(len(frames_ddpg), len(frames_erl))
max_g = min(len(games_ddpg), len(games_erl))
assert max_f == max_g # should tell if something is a bit off 

frames_ddpg = frames_ddpg[:max_f]; games_ddpg = games_ddpg[:max_f]
frames_erl = frames_erl[:max_f]; games_erl = games_erl[:max_f]
ddpg_score = ddpg_score[:max_f]; ddpg_std = ddpg_std[:max_f]
erl_score = erl_score[:max_f]; erl_std = erl_std[:max_f]



# Plotting:
do_plot = True
if do_plot:
    f2e = 10**4   # frames to epochs contraction for better plotting
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
    
    # Frames plot
    fig1,ax = plt.subplots()
    fig1.canvas.manager.set_window_title("Reward versus frames")
    # ax.set_title("Average Reward - LunarLander", pad=20)
    ax.plot(games_ddpg, erl_score[:,1], label = 'PD-ERL', color=colors[1])
    ax.plot(games_ddpg, ddpg_score[:,1], label = 'DDPG', color = colors[0])
    ax.fill_between(games_ddpg, erl_score[:,1]- erl_std[:,1], erl_score[:,1]+ erl_std[:,1], color=colors[1], alpha=0.4)
    ax.fill_between(games_ddpg, ddpg_score[:,1]- ddpg_std[:,1], ddpg_score[:,1]+ ddpg_std[:,1], color=colors[0],alpha=0.4)
    ax.set_ylabel("Reward [-]")
    ax.set_xlabel(r"Games [-]")
    ax.legend(loc = 'lower right')
    fig1.tight_layout()
    fig1.savefig('Results_pderl/Plots/reward_games.png')

    fig2,ax = plt.subplots()
    fig2.canvas.manager.set_window_title("Reward versus games")
    # ax.set_title("Average Reward - LunarLander", pad=20)
    ax.plot(frames_ddpg//f2e, erl_score[:,1], label = 'PD-ERL', color=colors[1])
    ax.plot(frames_ddpg//f2e, ddpg_score[:,1], label = 'DDPG', color=colors[0])
    ax.fill_between(frames_ddpg//f2e, erl_score[:,1]- erl_std[:,1], erl_score[:,1]+ erl_std[:,1], color=colors[1], alpha=0.4)
    ax.fill_between(frames_ddpg//f2e, ddpg_score[:,1]- ddpg_std[:,1], ddpg_score[:,1]+ ddpg_std[:,1], color=colors[0],alpha=0.4)
    ax.set_ylabel("Reward [-]")
    ax.set_xlabel(r"Epochs [$10^4$ frames]")
    ax.legend(loc = 'lower right')
    fig2.tight_layout()
    fig2.savefig('Results_pderl/Plots/reward_frames.png')
    plt.show()
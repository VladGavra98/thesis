from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re


# plot style
style = 'seaborn-darkgrid'
plt.style.use(style.lower()) 
plt.rcParams.update({'font.size': 12})
# plt.rcParams['figure.dpi'] = 200
# plt.rcParams['figure.figsize'] = [6, 5]   #disabled for better figures

# colours
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
color_ddpg = colors[0] if 'seaborn-darkgrid' in style else colors[2]
color_erl  = colors[3] if 'seaborn-darkgrid' in style else colors[0]
c_nominal  = colors[0] if 'seaborn-darkgrid' in style else colors[1]
c_fault1   = '#FBC15E' if 'seaborn-darkgrid' in style else colors[4]
c_fault2   = colors[4] if 'seaborn-darkgrid' in style else colors[5]

# Globals:
savefig = True
# nice purple: '#988ED5'

# Load state history data:
# logfolder = Path('/home/vlad/Documents/thesis/logs/wandb/run-20220706_104451-2qf6jy26/files/')
logfolder = Path('./logs/wandb/run-20220717_213834-nbmdpqgz')
logfolder = logfolder / Path('files/')

def plot_epiosde_lon(flst, ep_num_lst, idx, name : str = None):
    
    flst = [flst[i] for i in np.argsort(ep_num_lst)]
    ep_num_lst = np.sort(ep_num_lst)

    episode_file = open(logfolder / Path(flst[idx]),encoding = 'utf-8')

    # episode_num = episode_file.readline().strip('# ')
    data = np.genfromtxt(episode_file, skip_header=1)

    ref_signals = data[:,:1]; u_lst = data[:,1:2]; x_lst = data[:,2:-1]; rewards = data[:,-1]
    dt = 0.01
    time = np.linspace(0., x_lst.shape[0] * dt, x_lst.shape[0])
    print(name + ' validation fitness: ' , sum(rewards))

    fig, axs = plt.subplots(3,2)
    fig.suptitle(name + f' actor: episdoe {ep_num_lst[idx]}')

    axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = r'$q$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$')
    axs[0,0].plot(time,ref_signals[:,0], linestyle = '--',label = r'$\theta_{ref}$', color = colors[0])
    axs[0,0].set_ylabel(r'$\theta~[deg],q~[deg/s]$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$')


    axs[1,0].plot(time,x_lst[:,3], label = r'$V$')
    axs[1,0].set_ylabel(r'$V~[m/s]$')

    axs[2,0].plot(time,x_lst[:,9])
    axs[2,0].set_ylabel(r'$H~[m]$')

    # plot actions
    axs[0,1].plot(time,np.rad2deg(u_lst[:,0]), linestyle = '-')
    axs[0,1].set_ylabel(r'$\delta_e~[deg]$')

    # axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = 'nz')
    axs[1,1].plot(time[:-1],rewards[:-1])
    axs[1,1].set_ylabel('Reward [-]'); axs[1,1].set_xlabel('Time [s]')

    for i in range(3):
        for j in range(2):
            axs[i,j].set_xlabel('Time [s]')
            axs[i,j].legend(loc = 'best')

    plt.tight_layout()

def plot_epiosde_full (flst, ep_num_lst, idx, name : str = None):
    
    flst = [flst[i] for i in np.argsort(ep_num_lst)]
    ep_num_lst = np.sort(ep_num_lst)

    episode_file = open(logfolder / Path(flst[idx]),encoding = 'utf-8')

    # episode_num = episode_file.readline().strip('# ')
    data = np.genfromtxt(episode_file, skip_header=1)

    ref_signals = data[:,:3]; u_lst = data[:,3:6]; x_lst = data[:,6:-1]; rewards = data[:,-1]
    dt = 0.01
    time = np.linspace(0., x_lst.shape[0] * dt, x_lst.shape[0])
    print(name + ' validation fitness: ' , sum(rewards))

    fig, axs = plt.subplots(4,2)
    fig.suptitle(name + f' actor: episdoe {ep_num_lst[idx]}')

    axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = r'$q$', linestyle = ':', color = colors[4])
    axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$', color = colors[1])
    axs[0,0].plot(time,ref_signals[:,0], linestyle = '--',label = r'$\theta_{ref}$', color = colors[0])
    axs[0,0].set_ylabel(r'$\theta~[deg],q~[deg/s]$')
    # axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$')


    axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), label = r'$p$', linestyle = ':', color = colors[4])
    axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = r'$\phi$', color = colors[1])
    axs[1,0].plot(time,ref_signals[:,1], linestyle = '--',label = r'$\phi_{ref}$', color = colors[0])
    axs[1,0].set_ylabel(r'$\phi~[deg],p~[deg/s]$')


    axs[2,0].plot(time,np.rad2deg(x_lst[:,0]), label = r'$r$',  linestyle = ':', color = colors[4])
    axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = r'$\beta$', color = colors[1])
    axs[2,0].plot(time,ref_signals[:,2], linestyle = '--',label = r'$\beta_{ref}$', color = colors[0])
    axs[2,0].set_ylabel(r'$\beta~[deg],r~[deg/s]$')


    axs[3,0].plot(time,x_lst[:,9])
    axs[3,0].set_ylabel(r'$H~[m]$')

    # plot actions
    axs[0,1].plot(time,100*np.rad2deg(u_lst[:,0]), linestyle = '-')
    axs[0,1].set_ylabel(r'$\delta_e~[deg]$')
    axs[1,1].plot(time,100*np.rad2deg(u_lst[:,1]), linestyle = '-')
    axs[1,1].set_ylabel(r'$\delta_a~[deg]$')
    axs[2,1].plot(time,100*np.rad2deg(u_lst[:,2]), linestyle = '-')
    axs[2,1].set_ylabel(r'$\delta_r~[deg]$')

    # axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = 'nz')
    axs[3,1].plot(time[:-1],rewards[:-1])
    axs[3,1].set_ylabel('Reward [-]'); axs[1,1].set_xlabel('Time [s]')

    for i in range(3):
        for j in range(2):
            axs[i,j].set_xlabel('Time [s]')
            axs[i,j].legend(loc = 'best')

    plt.tight_layout()
if __name__ == '__main__':
    flst,rl_flst ,ep_num_lst, rl_ep_num_lst = [], [], [], []

    for file in os.listdir(logfolder):
        if file.endswith(".txt") and 'requirements' not in file:
            ep_num = int(re.search(r'\d+', file).group())
            if 'rl' in file:
                rl_flst.append(file)
                rl_ep_num_lst.append(ep_num)
            else:
                flst.append(file)
                ep_num_lst.append(ep_num)


    idx = -1
    # lon control episdoes 
    # if len(flst):
    #     plot_epiosde_lon(flst, ep_num_lst, idx, name = 'Champion')
    # plot_epiosde_lon(rl_flst, rl_ep_num_lst, idx, name = 'RL')

    # full attitude control episodes
    if len(flst):
        plot_epiosde_full(flst, ep_num_lst, idx, name = 'Champion')
    plot_epiosde_full(rl_flst, rl_ep_num_lst, idx, name = 'RL')


    plt.show()

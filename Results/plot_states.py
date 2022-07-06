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
logfolder = Path('/home/vlad/Documents/thesis/logs/wandb/latest-run/files/')


def plot_epsiode_data_champ(flst, ep_num_lst, idx):
    flst = [flst[i] for i in np.argsort(ep_num_lst)]
    episode_file = open(logfolder / Path(flst[idx]),encoding = 'utf-8')

    # episode_num = episode_file.readline().strip('# ')
    data = np.genfromtxt(episode_file, skip_header=1)

    ref_signals = data[:,:3]; u_lst = data[:,3:6]; x_lst = data[:,6:-1]; rewards = data[:,-1]
    dt = 0.01
    time = np.linspace(0., x_lst.shape[0] * dt, x_lst.shape[0])

    fig, axs = plt.subplots(4,2)
    fig.suptitle(f'Episdoe {ep_num_lst[idx]}')


    # axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = r'$q$')
    axs[0,0].plot(time,ref_signals[:,0], linestyle = '--',label = r'$\theta_{ref}$')
    axs[0,0].set_ylabel(r'$\theta~[deg],q~[deg/s]$')

    axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = r'$\phi$')
    axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), label = r'$p$')
    axs[1,0].plot(time,ref_signals[:,1], linestyle = '--',label = r'$\phi_{ref}$')
    axs[1,0].set_ylabel(r'$\phi~[deg],p~[deg/s]$')

    axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = r'$\beta$')
    axs[2,0].plot(time,ref_signals[:,2], linestyle = '--',label = r'$\beta_{ref}$')
    axs[2,0].set_ylabel(r'$\beta~[deg]$')

    axs[3,0].plot(time,x_lst[:,9])
    axs[3,0].set_ylabel(r'$H~[m]$')

    # plot actions
    axs[0,1].plot(time,np.rad2deg(u_lst[:,0]), linestyle = '-',label = r'$\delta_e$')
    axs[1,1].plot(time,np.rad2deg(u_lst[:,1]), linestyle = '-',label = r'$\delta_a$')
    axs[2,1].plot(time,np.rad2deg(u_lst[:,2]), linestyle = '-',label = r'$\delta_r$')
    # axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = 'nz')
    axs[3,1].plot(time,rewards)
    axs[3,1].set_ylabel('Reward [-]'); axs[3,1].set_xlabel('Time [s]')


    print('Validation fitness of champion: ' , sum(rewards))
    for i in range(4):
        for j in range(2):
            axs[i,j].set_xlabel('Time [s]')
            axs[i,j].legend(loc = 'best')

    plt.tight_layout()

def plot_epsiode_data_rl(flst, ep_num_lst, idx):
    flst = [flst[i] for i in np.argsort(ep_num_lst)]
    episode_file = open(logfolder / Path(flst[idx]),encoding = 'utf-8')

    # episode_num = episode_file.readline().strip('# ')
    data = np.genfromtxt(episode_file, skip_header=1)

    ref_signals = data[:,:1]; u_lst = data[:,1:2]; x_lst = data[:,2:-1]; rewards = data[:,-1]
    dt = 0.01
    time = np.linspace(0., x_lst.shape[0] * dt, x_lst.shape[0])

    fig, axs = plt.subplots(3,2)
    fig.suptitle(f'Episdoe {ep_num_lst[idx]}')

    # axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = r'$q$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$')
    axs[0,0].plot(time,ref_signals[:,0], linestyle = '--',label = r'$\theta_{ref}$')
    axs[0,0].set_ylabel(r'$\theta~[deg],q~[deg/s]$')


    axs[1,0].plot(time,x_lst[:,3], label = r'$V$')
    axs[1,0].set_ylabel(r'$V~[m/s]$')

    axs[2,0].plot(time,x_lst[:,9])
    axs[2,0].set_ylabel(r'$H~[m]$')

    # plot actions
    axs[0,1].plot(time,np.rad2deg(u_lst[:,0]), linestyle = '-',label = r'$\delta_e$')
   
    # axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = 'nz')
    axs[1,1].plot(time[:-1],rewards[:-1])
    axs[1,1].set_ylabel('Reward [-]'); axs[1,1].set_xlabel('Time [s]')


    print('Validation fitness of champion: ' , sum(rewards))
    for i in range(3):
        for j in range(2):
            axs[i,j].set_xlabel('Time [s]')
            axs[i,j].legend(loc = 'best')

    plt.tight_layout()

if __name__ == '__main__':
    flst,rl_flst , ep_num_lst = [], [], []

    for file in os.listdir(logfolder):
        if file.endswith(".txt") and 'requirements' not in file:
            ep_num = int(re.search(r'\d+', file).group())
            ep_num_lst.append(int(int(ep_num)))

            if 'rl' in file:
                rl_flst.append(file)
            else:
                flst.append(file)

    ep_num_lst = sorted(ep_num_lst)
    print(ep_num_lst)

    idx = -1
    if len(flst):
        plot_epsiode_data_champ(flst, ep_num_lst, idx)

    plot_epsiode_data_rl(rl_flst, ep_num_lst, idx)

    plt.show()

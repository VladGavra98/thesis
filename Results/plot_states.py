from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

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
f_lst = []
for file in os.listdir(logfolder):
    if file.endswith(".txt") and 'requirements' not in file:
        f_lst.append(file)

episode_file = open(logfolder / Path(f_lst[-1]),encoding = 'utf-8')
episode_num = episode_file.readline().strip('# ')
data = np.genfromtxt(episode_file, skip_header=1)

ref_signals = data[:,:3]; u_lst = data[:,3:6]; x_lst = data[:,6:]
dt = 0.01
time = np.linspace(0., x_lst.shape[0] * dt, x_lst.shape[0])

fig, axs = plt.subplots(4,2)
fig.suptitle('Episdoe ' + episode_num)
axs[0,0].plot(time,ref_signals[:,0], linestyle = '--',label = 'ref_theta')
axs[1,0].plot(time,ref_signals[:,1],linestyle = '--' ,label = 'ref_phi')
axs[2,0].plot(time,ref_signals[:,2], linestyle = '--',label = 'ref_beta')

axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = 'alpha')
axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = 'q')
axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = 'theta')

axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = 'beta')
axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = 'phi')
axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), label = 'p')
axs[3,0].plot(time,x_lst[:,9], label = 'H')


# plot actions
axs[0,1].plot(time,np.rad2deg(u_lst[:,0]), linestyle = '-',label = r'$\delta_e$')
axs[1,1].plot(time,np.rad2deg(u_lst[:,1]), linestyle = '-',label = r'$\delta_a$')
axs[2,1].plot(time,np.rad2deg(u_lst[:,2]), linestyle = '-',label = r'$\delta_r$')
# axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = 'nz')

# fig2, ax_reward = plt.subplots()
# # ax_reward.plot(time,rewards)
# ax_reward.set_ylabel('Reward [-]')
for i in range(4):
    for j in range(2):
        axs[i,j].set_xlabel('Time [s]')
        axs[i,j].legend(loc = 'best')

plt.tight_layout()
plt.show()

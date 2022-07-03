from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# plot style
style = 'seaborn-darkgrid'
plt.style.use(style.lower()) 
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 200
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

filename = f_lst[-1]
data = np.genfromtxt(logfolder / Path(filename), skip_header=1)

ref_sginals = data[:3]
u_lst = data[3:6]
x_lst = data[6:]
dt = 0.01
time = np.arange(0., x_lst.shape[0] * dt, dt)

fig, axs = plt.subplots(4,2)
# axs[0,0].plot(time,ref_theta, linestyle = '--',label = 'ref_theta')
# axs[1,0].plot(time,ref_phi,linestyle = '--' ,label = 'ref_phi')
# axs[2,0].plot(time,ref_beta, linestyle = '--',label = 'ref_beta')

axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = 'alpha')
axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = 'q')
axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = 'theta')

axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = 'beta')
axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = 'phi')
axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), label = 'p')
axs[3,0].plot(time,x_lst[:,9], label = 'H')


# plot actions
# axs[0,1].plot(time,u_lst[:,0], linestyle = '--',label = 'de')
# axs[1,1].plot(time,u_lst[:,1], linestyle = '--',label = 'da')
# axs[2,1].plot(time,u_lst[:,2], linestyle = '--',label = 'dr')
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

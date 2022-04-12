import numpy as np

import matplotlib.pyplot as plt

import os


logs = 'pderl/logs'
ddpg_score = np.genfromtxt(logs + '/ddpg_score.csv', skip_header= 1, delimiter=',')
erl_score = np.genfromtxt(logs + '/erl_score.csv', skip_header= 1, delimiter=',')

fig,ax = plt.subplots()
fig.suptitle("Average Reward for 'Swimmer v3'")
ax.plot(ddpg_score[:,0], ddpg_score[:,1], label = 'DDPG')
ax.plot(ddpg_score[:,0], erl_score[:,1], label = 'PD-ERL')
ax.set_ylabel("Average Reward [-]")
ax.set_xlabel("Epochs [frames]")
ax.legend(loc= 'best')
fig.tight_layout()
plt.show()
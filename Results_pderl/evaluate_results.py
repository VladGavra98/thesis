import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

logs = 'pderl/logs'
ddpg_score = np.genfromtxt(logs + '/ddpg_score.csv', skip_header= 1, delimiter=',')
erl_score = np.genfromtxt(logs + '/erl_score.csv', skip_header= 1, delimiter=',')

fig,ax = plt.subplots()
ax.set_title("Average Reward - LunarLander", pad=20)
ax.plot(ddpg_score[:,0]/10**5, ddpg_score[:,1], label = 'DDPG')
ax.plot(ddpg_score[:,0]/10**5, erl_score[:,1], label = 'PD-ERL')
ax.set_ylabel("Average Reward [-]")
ax.set_xlabel(r"Epochs [$10^5$ frames]")
ax.legend(loc = 'best')
fig.tight_layout()
plt.show()
import numpy as np
import os
import random


import matplotlib.pyplot as plt


# my envs
from envs.lunarlander import simulate

style = 'seaborn-darkgrid'
plt.style.use(style.lower()) 
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 200

name = 'comparative plot'

bcs_map_old = np.random.randn(30,2)
bcs_map = np.random.randn(30,2)
rewards = 500* np.random.randn(30)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.scatter(bcs_map[:, 0], bcs_map[:, 1],
                    c=rewards, marker='s', cmap='magma', s= 20)
u = bcs_map[:, 0] - bcs_map_old[:, 0]
v = bcs_map[:, 1] - bcs_map_old[:, 1]
img= ax.quiver(bcs_map[:, 0], bcs_map[:, 1],u,v, alpha = 0.5, pivot = 'tip', width = 0.003)
if name is not None:
    fig.suptitle('Archive: ' + str(name))

ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
ax.set_ylabel(r"Impact $\dot{y}$")
ax.set_xlabel(r"Impact $x$")

cbar = fig.colorbar(img, orientation='vertical')
cbar.ax.set_title('Mean Reward')
plt.tight_layout()

plt.show()
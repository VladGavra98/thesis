import numpy as np
import random
import matplotlib.pyplot as plt


style = 'seaborn-darkgrid'
plt.style.use(style.lower()) 
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 200

name = 'comparative plot'

bcs_map_old = np.random.randn(30,2)
bcs_map = np.random.randn(30,2)
rewards = 500* np.random.randn(30)
u = bcs_map[:, 0] - bcs_map_old[:, 0]
v = bcs_map[:, 1] - bcs_map_old[:, 1]


fig, ax = plt.subplots(figsize=(8, 6))
# draw arrows
width = 0.05
for i in range(len(bcs_map[:,0])):
    ax.arrow(bcs_map_old[i, 0], bcs_map_old[i, 1],u[i],v[i], head_width = width, color = (0,0,0,0.6), length_includes_head = True)

img = ax.scatter(bcs_map[:, 0], bcs_map[:, 1],
                    c=rewards, marker='s', cmap='magma', s= 20)
img = ax.scatter(bcs_map_old[:, 0], bcs_map_old[:, 1],
                    c=rewards, marker='o', cmap='magma', s= 15, alpha = 0.75)



assert bcs_map[:, 0].all() == (bcs_map_old[:, 0] + u).all()

ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
ax.set_ylabel(r"Impact $\dot{y}$")
ax.set_xlabel(r"Impact $x$")

cbar = fig.colorbar(img, orientation='vertical')
cbar.ax.set_title('Mean Reward')



if name is not None:
    fig.suptitle('Archive: ' + str(name))
plt.tight_layout()

plt.show()
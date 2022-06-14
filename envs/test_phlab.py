import numpy as np
import citation as model
import matplotlib.pyplot as plt
import time
model.initialize()


input = np.array([0.0,0.,0.,0.,0.,0.,0.,0.,0., 0.])

# Output legend:
#   0. p
#   1. q
#   2. r
#   3. vtas
#   4. alpha
#   5. beta
#   6. phi
#   7. theta
#   8. psi
#   9. he
#   10. xe
#   11. ye


states , times = [], []
count = 0
dt = 0.01 # do not change

max_steps = 10**6

t0 = time.time()
for i in range(max_steps):
    count+=1
    input[0] = -0.1/57.3 
    output = model.step(input)
    states.append(output)
    times.append(i *dt)
print(f'Totoal time taken for {i} steps: {time.time()- t0 : 0.2f} s')

print("Output shape:", states[-1].shape)

model.terminate()

states = np.asanyarray(states)

# Plotting
fig, axs = plt.subplots(2)
axs[0].plot(times,states[:,1])
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('q [rad/s]')
axs[1].plot(times,states[:,9], label = 'h [m]')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('h [m]')
plt.show()
print('done!')
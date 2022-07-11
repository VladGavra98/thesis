import numpy as np
rewards, bcs_lst, lengths = [], [], [] 
i = 0

pop_size = 10


for net in range(pop_size):   
    for _ in range(3):
        rewards.append(i)
        i+=1


print('Original rewards:', rewards)
# take average stats
rewards = np.asarray(rewards).reshape((-1,pop_size))


avg_score = np.mean(rewards, axis = 0)

print('Reshape rewards: ', rewards)
print('Avg_score:', avg_score)

x = 1
y =4
if x > 2 or y  < 5:
    print('Done')

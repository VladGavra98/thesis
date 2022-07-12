import numpy as np
rewards, bcs_lst, lengths = [], [], [] 
i = 0

pop_size = 10
lst = []
for i in range(10):
    u = np.random.randn(5)
    lst.append(u)
lst = np.asarray(lst)

print(lst)
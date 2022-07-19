import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(7)
SD = 2
a = np.random.randn(10) * SD


# # Novelty:
# def get_novelty(a):
#     N = np.zeros((a.shape[0]))
#     A = a.shape[0]
#     for i in range(a.shape[0]):
#         N[i] = 1/A * np.sum(np.linalg.norm(a - a[i]))
#     # print(f'Novelty of population: { : 0.3f}')
#     N = np.sum(N)/A
#     N2 = np.sum(np.std(a, axis = 0))/a.shape[1]
#     # print(f'SD estimate: {(N2):0.3f}')

#     return N, N2

# N_base, N2_base = get_novelty(a)
# for _ in range(10):
#     # copy vector
#     cur_a = a[:]
#     # infuse a very novel behaviour
#     mutation = np.random.randint(2,5)
#     cur_a[np.random.randint(0,cur_a.shape[0]-1)] *= mutation
#     N, N2 = get_novelty(cur_a)
#     print(f'Sensitivity: M1: {(N-N_base)/mutation:0.3f}      M2: {(N2- N2_base)/mutation:0.3f}')




a = np.array([1,-1,1])
b = np.array([2,3,4])

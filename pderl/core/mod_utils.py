
from torch.autograd import Variable
import random, pickle
import numpy as np
import torch
import os
import torch.nn as nn

class Tracker:
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string
        self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [ [[],0.0,[]] for _ in vars_string] # [Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = 10
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        """ Update the logger.
        Args:
            updates (list): Variables to add to logger.
            generation (int): Counter (time passed/ frames/ games)
        """
        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        # Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        # Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                filename = os.path.join(self.foldername, self.vars_string[i] + self.project_string)
                try:
                    np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')
                except:
                    # Common error showing up in the cluster for unknown reasons
                    print('Failed to save progress')


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                            # Update Fucntions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                            # Noise
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GaussianNoise:

    def __init__(self, action_dimension, sd=0.1, mu=0):
        self.action_dimension = action_dimension
        self.sd= sd
        self.mu = mu

        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        return np.random.normal(self.mu,self.sd,self.action_dimension)


class OUNoise:

    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                            # Convertors
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def to_numpy(var):
    return var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def pickle_obj(filename, object):
    handle = open(filename, "wb")
    pickle.dump(object, handle)

def unpickle_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def odict_to_numpy(odict):
    l = list(odict.values())
    state = l[0]
    for i in range(1, len(l)):
        if isinstance(l[i], np.ndarray):
            state = np.concatenate((state, l[i]))
        else: #Floats
            state = np.concatenate((state, np.array([l[i]])))
    return state

def min_max_normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return (x - min_x) / (max_x - min_x)

def is_lnorm_key(key):
    return key.startswith('lnorm')




def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    #v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

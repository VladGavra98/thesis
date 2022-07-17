import torch
from torch import nn as nn

# Without Learnable Parameters
m = nn.InstanceNorm1d(100)
# With Learnable Parameters
m = nn.InstanceNorm1d(100, affine=True)
input = torch.randn(1, 100, 2)
output = m(input)

print(output.shape)
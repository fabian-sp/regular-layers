"""
@author: Fabian Schaipp
"""

import torch
from layers import L1Linear

U = torch.randn(4,5)
V = torch.randn(4,5)

U
V
U.mul(V)
#%%
b = 5 # batch size
n = 10 # input size
m = 3 # output size

X = torch.randn(b,n)

M = L1Linear(l1=0.1, in_features=n, out_features=m, bias=False)

M.weight_u


M(X)
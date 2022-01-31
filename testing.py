"""
@author: Fabian Schaipp
"""

import torch
import matplotlib.pyplot as plt

from layers import L1Linear
from lasso import TorchLasso

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

#%%

p = 20 # variables
N = 100 # samples
k = 5 # nonzeros
noise = 0.01

# oracle
beta = torch.concat((torch.zeros(p-k,1),torch.randn(k,1)))
beta = beta[torch.randperm(p)]

X = torch.randn(N,p)
y = X @ beta + noise*torch.randn(N,1)


l1 = 0.0001
batch_size = 10

model, info, iterates = TorchLasso(X, y, l1, bias=False, n_epochs=30, batch_size=10, store_iterates=True)

sol = model.get_weight()

#%%

fig, ax = plt.subplots()

ax.plot(info['train_loss'], label = 'Objective')
ax.plot(info['lsq_loss'], label = 'Residual loss')
ax.plot(info['reg'], label = 'Regularization term')

ax.set_xlabel('Epoch')
ax.legend()

#%%

fig, ax = plt.subplots()
ax.plot(iterates)
ax.set_xlabel('Epoch')
"""
@author: Fabian Schaipp
"""

import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

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

#%% solve Lasso with Pytorch

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

model, info, iterates = TorchLasso(X, y, l1, bias=False, n_epochs=50, batch_size=10, store_iterates=True)

sol = model.get_weight()

#%% compute Lasso path

alphas, coef_path, _ = lasso_path(X.numpy(), y.numpy().reshape(-1), alphas=None)

#%%

fig, ax = plt.subplots()

ax.plot(info['train_loss'], label = 'Objective')
ax.plot(info['lsq_loss'], label = 'Residual loss')
ax.plot(info['reg'], label = 'Regularization term')

ax.set_xlabel('Epoch')
ax.legend()

#%%

fig, axs = plt.subplots(1,2)
axs[0].plot(iterates)
axs[0].set_xlabel('Epoch')

axs[1].plot(alphas, coef_path.T, '-o', markersize = 0.8)
axs[1].set_xscale('log')
axs[1].set_xlabel('log(Lambda)')
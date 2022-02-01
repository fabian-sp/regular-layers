"""
@author: Fabian Schaipp
"""

import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, Lasso

from layers import L1Linear
from lasso import TorchLasso


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
noise = 0.00


# oracle
beta = torch.concat((torch.zeros(p-k,1),torch.randn(k,1)))
beta = beta[torch.randperm(p)]

X = torch.randn(N,p)
y = X @ beta + noise*torch.randn(N,1)

#compute Lasso path
alphas, coef_path, _ = lasso_path(X.numpy(), y.numpy().reshape(-1), alphas=None)

# lasso path uses no intercept!
#sk = Lasso(alphas[-1], fit_intercept=False).fit(X.numpy(), y.numpy())
#sk.coef_
#coef_path[:,-1]

#%%

l1 = 2*alphas.min() # factor 2 comes from diff. obj in scikit
batch_size = 10

model, info, iterates = TorchLasso(X, y, l1, bias=False, n_epochs=110, lr = 0.01, batch_size=batch_size, store_iterates=True)

sol = model.get_weight()

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
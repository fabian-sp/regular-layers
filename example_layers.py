"""
@author: Fabian Schaipp
"""

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from layers import L1Linear, LowRankLinear

#%%
b = 5 # batch size
n = 10 # input size
m = 1 # output size
N = 1000 # samples

X = torch.randn(b,n)

def generate_toy_example(N):
    X = torch.randn(N, n)  
    y = 2.*X[:, 3] - 1.*X[:, 3]**2 + 1.*X[:, 1] + 0.5*X[:, 2] + 2 * X[:, 4] * X[:, 5]
    return X, y.reshape(-1,1)

X, Y = generate_toy_example(N)

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx, :]
        y = self.targets[idx]
        return x, y
    
ds = MyDataset(X,Y)
dl = DataLoader(ds, batch_size=b, shuffle=True)

#%%
class ExampleNetwork(torch.nn.Module):
    """
    2-layer NN with RelU
    first layer sparse, second layer low rank
    """
    def __init__(self, l1=0.):
        super().__init__()
        self.W1 = L1Linear(l1=l1, in_features=n, out_features=30, bias=False)
        self.relu = torch.nn.ReLU()
        self.W2 = LowRankLinear(l1=l1, in_features=30, out_features=30, bias=False)
        self.W3 = torch.nn.Linear(in_features=30, out_features=m, bias=True)
        return
    
    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        x = self.relu(x)
        x = self.W3(x)
        return x
    
l1 = 0.1
model = ExampleNetwork(l1=l1)
loss = torch.nn.MSELoss(reduction='mean')
opt = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=l1, momentum=0.9, nesterov=True)

#%%
n_epochs = 30

for j in torch.arange(n_epochs): 
    for inputs, targets in dl:
        # forward pass
        y_pred = model.forward(inputs)
        # compute loss
        loss_val = loss(y_pred, targets)           
        # zero gradients
        opt.zero_grad()    
        # backward pass
        loss_val.backward()    
        # iteration
        opt.step()

    print(loss_val.item())        
        
model.W1.get_weight()
model.W2.get_weight()
torch.linalg.matrix_rank(model.W2.get_weight())

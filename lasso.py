"""
@author: Fabian Schaipp
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from layers import L1Linear

class LassoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return x, y
    

def TorchLasso(X: torch.tensor, y: torch.tensor, l1: float, opt: torch.optim.Optimizer = None, bias: bool = True,\
               verbose: bool = False, n_epochs: int = 2, batch_size: int = 1, store_iterates: bool = False):
    """

    Parameters
    ----------
    X : torch.tensor
        DESCRIPTION.
    y : torch.tensor
        DESCRIPTION.
    l1 : float
        DESCRIPTION.
    opt : torch.optim.Optimizer, optional
        Optimizer. The default is SGD.
    bias : bool, optional
        DESCRIPTION. The default is True.
    verbose : bool, optional
        DESCRIPTION. The default is False.
    n_epochs : int, optional
        DESCRIPTION. The default is 2.
    batch_size : int, optional
        DESCRIPTION. The default is 1.
    store_iterates : bool, optional
        Whether to store the iterates. The default is False.

    Returns
    -------
    model : torch.nn.Module
        Trained module.
    info : dict
        History of total loss, resiudal and penalty term values.

    """
    p = X.shape[1]
    N = X.shape[0]
    
    ds = LassoDataset(X, y)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = False)
    
    dataiter = iter(dl)
    inputs, targets = dataiter.next()

    model = L1Linear(l1=l1, in_features=p, out_features=1)
    
    if opt is None:
        opt = torch.optim.SGD(model.parameters(), lr = 0.01)
        
    iterates = list()   
    info = {'train_loss':[], 'lsq_loss':[], 'reg':[]}
    
    
    loss = torch.nn.MSELoss()
    
    for j in torch.arange(n_epochs):
        ################### SETUP FOR EPOCH ##################
        all_loss = list(); all_lsq = list(); all_reg = list()
        ################### START OF EPOCH ###################
        model.train()
        for inputs, targets in dl:
            
            # forward pass
            y_pred = model.forward(inputs)
            # compute loss
            lsq = loss(y_pred, targets)
            pen = model.reg()
            loss_val = lsq + pen            
            # zero gradients
            opt.zero_grad()    
            # backward pass
            loss_val.backward()    
            # iteration
            opt.step()
            
            all_loss.append(loss_val.item())
            all_lsq.append(lsq.item())
            all_reg.append(pen.item())
            
            if store_iterates:
                iterates.append(model.get_weight())
        ################### END OF EPOCH ###################
        
        ### STORE
        info['train_loss'].append(np.mean(all_loss))
        info['lsq_loss'].append(np.mean(all_lsq))
        info['reg'].append(np.mean(all_reg))
        
        if verbose:
            print(f"Epoch {j+1}/{n_epochs}: \t  train loss: {np.mean(all_loss)}.")
            print(opt)    
    
    if store_iterates:
        iterates = torch.concat(iterates).detach().numpy()
    
    return model, info, iterates

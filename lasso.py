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
    

def TorchLasso(X: torch.tensor, y: torch.tensor, l1: float, opt: torch.optim.Optimizer=None, bias: bool=True,\
               verbose: bool=False, n_epochs: int=2, lr:float=0.01, batch_size: int=1, store_iterates: bool=False):
    """
    For :math:`X\\in\\mathbb{R}^{N\\times p}`, solves (if bias=True)
    
        min_{w,w0}  1/N * ||Xw + w0 -y||^2 + l1*||w||_1
        
    Parameters
    ----------
    X : torch.tensor
        Coefficient matrix.
    y : torch.tensor
        Targets.
    l1 : float
        Regularization parameter.
    opt : torch.optim.Optimizer, optional
        Optimizer. The default is SGD.
    bias : bool, optional
        Allow bias for layer. The default is True.
    verbose : bool, optional
        Verbosity. The default is False.
    n_epochs : int, optional
        Number of epochs. The default is 2.
    lr : float, optional
        learning rate. The default is 1e-2
    batch_size : int, optional
        batch size. The default is 1.
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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    model = L1Linear(l1=l1, in_features=p, out_features=1)
    
    # weight_decay/2 * (||u||^2+||v||^2) 
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l1)
    
    iterates = list()   
    info = {'train_loss':[], 'lsq_loss':[], 'reg':[], 'tol':[]}
    
    loss = torch.nn.MSELoss()
    
    for j in torch.arange(n_epochs):
        ################### SETUP FOR EPOCH ##################
        all_loss = list(); all_lsq = list(); all_reg = list(); all_tol = list()
        ################### START OF EPOCH ###################
        model.train()
        for inputs, targets in dl:
            
            # zero gradients
            opt.zero_grad()    
            # forward pass
            y_pred = model.forward(inputs)
            # compute loss
            lsq = loss(y_pred, targets) # =1/N ||Xw-y||^2
            # backward pass
            lsq.backward()    
            # iteration
            opt.step()
            
            with torch.no_grad():
                pen = model.reg() # = l1 ||w||_1
            
            all_loss.append(lsq.item()+pen.item())
            all_lsq.append(lsq.item())
            all_reg.append(pen.item())
            all_tol.append(model.get_tol().item())
            
            
        ################### END OF EPOCH ###################
        if store_iterates:
            iterates.append(model.get_weight())
                
        ### STORE
        info['train_loss'].append(np.mean(all_loss))
        info['lsq_loss'].append(np.mean(all_lsq))
        info['reg'].append(np.mean(all_reg))
        info['tol'].append(np.mean(all_tol))
        
        if verbose:
            print(f"Epoch {j+1}/{n_epochs}: \t  train loss: {np.mean(all_loss)}.")
            print(opt)    
    
    if store_iterates:
        iterates = torch.concat(iterates).detach().numpy()
    
    return model, info, iterates

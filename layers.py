"""
@author: Fabian Schaipp

using the conventions from torch.Linear, see 
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
"""
import warnings

import math
import torch
from torch.nn.parameter import Parameter

from torch import Tensor
from torch.nn import functional as F
from torch.nn import init

class L1Linear(torch.nn.Module):
    

    def __init__(self, l1: float, in_features: int, out_features: int, bias: bool=True,
                 init_zero=False, device=None, dtype=None) -> None:
        """Applies a linear transformation to the incoming data: :math:`y = xW^T + b`
           where the weights of :math:`W` are l1-penalized, i.e. with the term
               
           l1 * ||W||_1
           
           The weights :math:`W` are never actually optimized but rather they are split up into two factors
           ``self.weight_u`` and ``self.weight_v``. The lweights :math:`W` can be retrieved by ``self.get_weight()``.
           For further details see
            
             Equivalences Between Sparse Models and Neural Networks, Ryan Tibshirani, 2021.

        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
        Args:
            l1: float, regularization parameter
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias. The bias is not regularized.
                Default: ``True``
        
        Shape:
            - Input: :math:`(*, H_{in})` where :math:`*` means any number of
              dimensions including none and :math:`H_{in} = \text{in\_features}`.
            - Output: :math:`(*, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    
        Attributes:
            weight_u: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized to zero.
            weight_v: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias: the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
            init_zero: whether to initialize one of the weights at zero.
        
        Examples::
    
            TBD
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(L1Linear, self).__init__()
        
        assert l1 >= 0, "l1 should be non-negative."
        if l1 == 0:
            warnings.warn("Choosing l1=0 means no regularization. You should use the standard Pytorch Linear module.")
        
        self.l1 = l1
        self.in_features = in_features
        self.out_features = out_features
        self.weight_u = Parameter(torch.zeros((out_features, in_features), **factory_kwargs))
        self.weight_v = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters(init_zero)
        
        return

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight_u.mul(self.weight_v), self.bias)
    
    ### Custom methods
    def reg(self):
        """
        compute l1 * ||W||_1 = (l1/2)* (||W_u||^2 + ||W_v||^2)
        """
        return (self.l1/2) * (torch.linalg.norm(self.weight_u)**2 + torch.linalg.norm(self.weight_u)**2)
    
    def get_weight(self):
        return self.weight_u.mul(self.weight_v)
    
    def get_tol(self):
        """ Stopping criterion for the split, i.e. at the minimizer we have ``abs(self.weight_u) == abs(self.weight_v)`` """
        return torch.max(torch.abs(torch.abs(self.weight_u)-torch.abs(self.weight_v)))
    
    ####
    def reset_parameters(self, init_zero=False) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        
        if not init_zero:
            init.kaiming_uniform_(self.weight_u, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_u)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def extra_repr(self) -> str:
        return 'l1={}, in_features={}, out_features={}, bias={}'.format(
            self.l1, self.in_features, self.out_features, self.bias is not None
        )
    
#%%

class LowRankLinear(torch.nn.Module):
    
    def __init__(self, l1: float, in_features: int, out_features: int, bias: bool=True,
                 init_zero=False, device=None, dtype=None) -> None:
        """Applies a linear transformation to the incoming data: :math:`y = xW^T + b`
           where the weights of :math:`W` are penalized with the nuclear norm, i.e. with the term
               
           l1 * ||W||_nuc
                   
           The weights :math:`W` are never actually optimized but rather they are split up into two factors
           ``self.weight_u`` and ``self.weight_v``. The lweights :math:`W` can be retrieved by ``self.get_weight()``.
           For further details see
            
             Equivalences Between Sparse Models and Neural Networks, Ryan Tibshirani, 2021.

        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
        Args:
            l1: float, regularization parameter
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias. The bias is not regularized.
                Default: ``True``
        
        Shape:
            - Input: :math:`(*, H_{in})` where :math:`*` means any number of
              dimensions including none and :math:`H_{in} = \text{in\_features}`.
            - Output: :math:`(*, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    
        Attributes:
            weight_u: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized to zero.
            weight_v: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias: the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
            init_zero: whether to initialize one of the weights at zero.
        
        Examples::
    
            TBD
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LowRankLinear, self).__init__()
        
        assert l1 >= 0, "l1 should be non-negative."
        if l1 == 0:
            warnings.warn("Choosing l1=0 means no regularization. You should use the standard Pytorch Linear module.")
        
        self.l1 = l1
        self.in_features = in_features
        self.out_features = out_features
        self._rank = min(in_features, out_features) 
        self.weight_u = Parameter(torch.zeros((out_features, self._rank), **factory_kwargs))
        self.weight_v = Parameter(torch.empty((in_features, self._rank), **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters(init_zero)
        
        return

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight_u @ self.weight_v.T, self.bias)
    
    ### Custom methods
    def reg(self):
        """
        compute l1 * ||W||_nuc = (l1/2)* (||W_u||^2 + ||W_v||^2)
        """
        return (self.l1/2) * (torch.linalg.norm(self.weight_u)**2 + torch.linalg.norm(self.weight_u)**2)
    
    def get_weight(self):
        return self.weight_u @ self.weight_v.T
    
    ####
    def reset_parameters(self, init_zero=False) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        
        if not init_zero:
            init.kaiming_uniform_(self.weight_u, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_u)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def extra_repr(self) -> str:
        return 'l1={}, in_features={}, out_features={}, bias={}'.format(
            self.l1, self.in_features, self.out_features, self.bias is not None
        )
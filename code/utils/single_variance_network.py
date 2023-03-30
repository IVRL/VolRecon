import torch 
from torch import nn 

# global variance as in SparseNeuS
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * torch.exp(self.variance * 10.0)

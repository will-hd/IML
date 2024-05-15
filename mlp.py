import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable

class MLP(nn.Module):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int,
                 num_h_layers: int, # number of hidden layers
                 bias: bool,
                 hidden_acc: Callable,
                 output_acc: Callable
                 ):
        super().__init__()

        h = [hidden_size] * (num_h_layers)

        layers = []
        for i, (h, k) in enumerate(zip([in_size] + h, h + [out_size])):
            layers.append(nn.Linear(h, k, bias=bias))
            if i != num_h_layers:
                layers.append(hidden_acc)
            else:
                layers.append(output_acc)

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(nn.Linear(i, o, bias=bias) for i, o in zip([in_size] + h, h + [out_size]))

    def forward(self, x):
#        for i, layer in enumerate(self.layers):
#            x = self.hidden_acc(layer(x)) if i != self.num_h_layers else self.output_acc(layer(x))
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":

    test = MLP(5, 5, 10, 2, bias=True, hidden_acc=nn.ReLU(), output_acc=nn.Sigmoid())
    print(test)

    x = torch.Tensor([1,2,3,4,5])

    print(test(x))

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable

class BatchMLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_h_layers: int,
                 use_bias: bool,
                 hidden_activation: Callable,
                 output_activation: Callable
                 ) -> None:
        super().__init__()
        
        layers = self.make_MLP_layers(input_dim=input_dim,
                                      output_dim=output_dim,
                                      hidden_dim=hidden_dim,
                                      n_h_layers=n_h_layers,
                                      use_bias=use_bias,
                                      hidden_activation=hidden_activation,
                                      output_activation=output_activation)
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.005)
            # nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.zero_()

    def _initialize_weights(self):
        for module in self.mlp:
            self._init_weights(module)


    @staticmethod
    def make_MLP_layers(input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_h_layers: int, # number of hidden layers
                 use_bias: bool,
                 hidden_activation: Callable,
                 output_activation: Callable
                 ) -> list[nn.Module]:

        h = [hidden_dim] * (n_h_layers)

        layers = []
        for i, (n, m) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(nn.Linear(n, m, bias=use_bias))
            if i != n_h_layers:
                layers.append(hidden_activation)
            else:
                layers.append(output_activation)

        return layers


# if __name__ == "__main__":

#     test = MLP(5, 5, 10, 2, bias=True, hidden_acc=nn.ReLU(), output_acc=nn.Sigmoid())
#     print(test)

#     x = torch.Tensor([1,2,3,4,5])

#     print(test(x))

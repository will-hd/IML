import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable

import logging
logger = logging.getLogger(__name__)

class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_h_layers: int,
                 use_bias: bool,
                 hidden_activation: nn.Module | None,
                 ) -> None:
        super().__init__()

        if n_h_layers == 0 and hidden_activation is not None:
            logging.warning(f'Hidden activation set but no hidden layers') 
        
        h = [hidden_dim] * (n_h_layers)
        layers = []
        for idx, (_in_dim, _out_dim) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(nn.Linear(_in_dim, _out_dim, bias=use_bias))
            if idx != n_h_layers: # add act to all but final layer
                layers.append(hidden_activation)

        self.mlp = nn.Sequential(*layers)

        # self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _initialize_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.005)
                # nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    module.bias.data.zero_()

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




if __name__ == "__main__":

    HIDDEN_DIM = 32 

    zero_hidden_mlp = MLP(input_dim=10,
                          output_dim=5,
                          hidden_dim=HIDDEN_DIM,
                          n_h_layers=0,
                          use_bias=True,
                          hidden_activation=nn.ReLU())
    print(zero_hidden_mlp)


    BATCH_SIZE = 32

    input_tensor = torch.randn(BATCH_SIZE, 10)
    output_tensor = zero_hidden_mlp(input_tensor)
    print(output_tensor.shape)

    one_hidden_mlp = MLP(input_dim=10,
                         output_dim=5,
                         hidden_dim=HIDDEN_DIM,
                         n_h_layers=1,
                         use_bias=True,
                         hidden_activation=nn.ReLU())
    print(one_hidden_mlp)

    output_tensor = one_hidden_mlp(input_tensor)
    print(output_tensor.shape)

    NUM_HIDDEN_LAYERS = 3
    hidden_mlp = MLP(input_dim=10,
                     output_dim=5,
                     hidden_dim=HIDDEN_DIM,
                     n_h_layers=NUM_HIDDEN_LAYERS,
                     use_bias=True,
                     hidden_activation=nn.ReLU())
    print(hidden_mlp)

    output_tensor = hidden_mlp(input_tensor)
    print(output_tensor.shape)



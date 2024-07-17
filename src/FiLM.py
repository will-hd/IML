import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable

class FiLMBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,  # Shape (batch_size, num_points, hidden_dim)
                gamma: torch.Tensor,  # Shape (batch_size, hidden_dim)
                beta: torch.Tensor  # Shape (batch_size, hidden_dim)
               ):
        
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        x = gamma * x + beta
        return x

class FiLM_MLP(nn.Module):

    def __init__(self,
                 x_input_dim: int,
                 k_input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_h_layers: int,
                 use_bias: bool,
                 hidden_activation: Callable,
                 output_activation: Callable
                 ) -> None:
        super().__init__()

        self.n_h_layers = n_h_layers
        self.hidden_dim = hidden_dim

        self.FiLM_generator = nn.Linear(k_input_dim, 2*hidden_dim*n_h_layers)
        
        self.FiLM_mlp = self.make_FiLM_MLP_layers(x_input_dim=x_input_dim,
                                      k_input_dim=k_input_dim,
                                      output_dim=output_dim,
                                      hidden_dim=hidden_dim,
                                      n_h_layers=n_h_layers,
                                      use_bias=use_bias,
                                      hidden_activation=hidden_activation,
                                      output_activation=output_activation)

        self._initialize_weights()

    def forward(self,
                x: torch.Tensor,
                film_input: torch.Tensor
               ) -> torch.Tensor:
        batch_size, _, _ = x.size()
        FiLM_params = self.FiLM_generator(film_input)
        FiLM_params = FiLM_params.view(batch_size, self.n_h_layers, self.hidden_dim, 2)

        hidden_dim_cnt = 0 
        for operation in self.FiLM_mlp:
            if isinstance(operation, FiLMBlock):
                gamma, beta = FiLM_params[:, hidden_dim_cnt, :, 0], FiLM_params[:, hidden_dim_cnt, :, 1]
                hidden_dim_cnt += 1
                x = operation(x, gamma, beta)
            else:
                x = operation(x)
        return x


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.001)
            if module.bias is not None:
                module.bias.data.zero_()

    def _initialize_weights(self):
        for module in self.FiLM_mlp:
            self._init_weights(module)


    @staticmethod
    def make_FiLM_MLP_layers(x_input_dim: int,
                        k_input_dim: int,
                        output_dim: int,
                        hidden_dim: int,
                        n_h_layers: int, # number of hidden layers
                        use_bias: bool,
                        hidden_activation: Callable,
                        output_activation: Callable
                       ) -> nn.ModuleList:
        module_list = nn.ModuleList()
        
        h = [hidden_dim] * (n_h_layers)

        for i, (n, m) in enumerate(zip([x_input_dim] + h, h + [output_dim])):
            module_list.add_module(f'Linear{i}', nn.Linear(n, m, bias=use_bias))
            if i != n_h_layers:
                module_list.add_module(f'Hidden_Activation{i}', hidden_activation)
                module_list.add_module(f'FiLMBlock{i}', FiLMBlock())
            else:
                module_list.add_module(f'Output_Activation', output_activation)


        return module_list

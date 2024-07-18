import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable

"""
TODO:
- Make FiLMBlock into a linear+film layers
- Put FiLmk
"""
class LinearFiLMActBlock(nn.Module):
    """
    Linear layer followed by FiLM layer followed by activation
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 use_bias: bool,
                 activation: nn.Module,
                 FiLM_before_activation: bool
                 ):
        super().__init__()
        self._FiLM_before_activation = FiLM_before_activation

        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.activation = activation

    def forward(self,
                x: torch.Tensor,  # Shape (batch_size, num_points, hidden_dim)
                gamma: torch.Tensor | None,  # Shape (batch_size, 1, hidden_dim)
                beta: torch.Tensor | None  # Shape (batch_size, 1, hidden_dim)
               ):
        x = self.linear(x)

        if self._FiLM_before_activation:
            if gamma is not None and beta is not None:
                # gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
                x = gamma * x + beta
            x = self.activation(x)
        else:
            x = self.activation(x)
            if gamma is not None and beta is not None:
                # gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
                x = gamma * x + beta
        return x

#class FiLMMLP(nn.Module):
#
#    def __init__(self,
#                 x_input_dim: int,
#                 k_input_dim: int,
#                 output_dim: int,
#                 hidden_dim: int,
#                 n_h_layers: int,
#                 use_bias: bool,
#                 hidden_activation: nn.Module,
#                 FiLM_before_activation: bool
#                 ) -> None:
#        super().__init__()
#
#        self.n_h_layers = n_h_layers
#        self.hidden_dim = hidden_dim
#
#        
#        module_list = nn.ModuleList()
#        
#        h = [hidden_dim] * (n_h_layers)
#
#        for idx, (_in_dim, _out_dim) in enumerate(zip([x_input_dim] + h, h + [output_dim])):
#            module_list.add_module(f'Linear{idx}', nn.Linear(_in_dim, _out_dim, bias=use_bias))
#            if idx != n_h_layers:
#                if FiLM_before_activation:
#                    module_list.add_module(f'FiLMBlock{idx}', FiLMBlock())
#                    module_list.add_module(f'Hidden_Activation{idx}', hidden_activation)
#                else:
#                    module_list.add_module(f'Hidden_Activation{idx}', hidden_activation)
#                    module_list.add_module(f'FiLMBlock{idx}', FiLMBlock())
#
#        self._initialize_weights()
#
#    def forward(self,
#                x: torch.Tensor,
#                film_input: torch.Tensor
#               ) -> torch.Tensor:
#        batch_size, _, _ = x.size()
#        FiLM_params = self.FiLM_params_generator(film_input)  # Shape (batch_size, 2*hidden_dim*n_h_layers) 
#        FiLM_params = torch.split(FiLM_params, 2*self.hidden_dim, dim=-1)
#
#        for 
#        # FiLM_params = FiLM_params.view(batch_size, self.n_h_layers, self.hidden_dim, 2)

#        hidden_dim_cnt = 0 
#        for operation in self.FiLM_mlp:
#            if isinstance(operation, FiLMBlock):
#                gamma, beta = FiLM_params[:, hidden_dim_cnt, :, 0], FiLM_params[:, hidden_dim_cnt, :, 1]
#                hidden_dim_cnt += 1
#                x = operation(x, gamma, beta)
#            else:
#                x = operation(x)
        return x


    def _initialize_weights(self):
        for module in self.FiLM_mlp:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.001)
                if module.bias is not None:
                    module.bias.data.zero_()

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from .attention import MultiheadAttender

from typing import Literal

import logging
logger = logging.getLogger(__name__)

class XEncoder(nn.Module):
    def __init__(self,
                 x_dim: int,
                 x_proj_dim: int,
                 hidden_dim: int,
                 n_h_layers: int,
                 activation: nn.Module,
                 use_bias: bool,
                 ):
        super().__init__()
        self.mlp = MLP(input_dim=x_dim,
                       output_dim=x_proj_dim,
                       hidden_dim=hidden_dim,
                       n_h_layers=n_h_layers,
                       use_bias=use_bias,
                       hidden_activation=activation)


    def forward(self, x):
        return self.mlp(x)


class XYSetEncoder(nn.Module):
    def __init__(self, 
                 x_dim: int,
                 y_dim: int,
                 hidden_dim: int,
                 n_h_layers_phi: int,
#                 n_h_layers_rho: int,
                 use_self_attn: bool,
                 n_self_attn_heads: int | None,
                 set_agg_function: Literal['mean'],
                 use_bias: bool,
                 x_target: None = None,
                 ):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of x
        """
        super().__init__()
        
        logging.debug(f'XYEncoder has x_dim={x_dim} and y_dim={y_dim}')
        
        self._use_self_attn = use_self_attn
        if n_self_attn_heads and not use_self_attn:
            logging.warning(f'n_self_attn_heads provided but use_self_attn is False')

        self._set_agg_function = set_agg_function

        self.phi = MLP(input_dim=x_dim+y_dim,
                            output_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_phi,
                            use_bias=use_bias,
                            hidden_activation=nn.GELU())

        if use_self_attn:
            assert n_self_attn_heads is not None, "n_self_attn_heads must be provided if use_self_attn is True"
            self.self_attender =  MultiheadAttender(
                    kq_size=hidden_dim,
                    value_size=hidden_dim,
                    out_size=hidden_dim,
                    is_post_process=True,
                    n_heads=n_self_attn_heads,
            )
        
#        if return_mean_repr and n_h_layers_rho > 0:
#            logging.info('XYEncoder is 
#            self.rho = MLP(input_dim=hidden_dim,
#                           output_dim=latent_dim,
#                           hidden_dim=hidden_dim,
        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        Returns
        -------
        r : torch.Tensor
            Encoding of the set
            Shape (batch_size, 1, hidden_dim)
            If x is empty, returns zeros
        """

        if x.shape[1] == 0: # Empty context set
            return torch.zeros((x.shape[0], 1, x.shape[-1])).to(x.device)
        else:
            xy_input = torch.cat((x, y), dim=-1)

            xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points, hidden_dim)
            
            if self._use_self_attn:
                xy_encoded = self.self_attender(xy_encoded, xy_encoded, xy_encoded) # Shape (batch_size, num_points, hidden_dim)

            if self._set_agg_function == 'mean':
                mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True)
            else:
                raise NotImplementedError(f'Aggregation function {self._set_agg_function} is not implemented')
            
            return mean_repr # Shape (batch_size, 1, hidden_dim)
#        mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)


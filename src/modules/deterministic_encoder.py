import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_modules import MultiheadAttention, CrossAttention
from .mlp import MLP

from typing import Literal

class DeterminisitcEncoder(nn.Module):
    """
    Deterministic encoder
    """
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dim: int,
                 determ_dim: int, # i.e. z_dim
                 n_h_layers_phi: int,
                 n_h_layers_rho: int,
                 use_cross_attn: bool,
                 use_self_attn: bool,
                 use_bias: bool
                 ):
        
        super().__init__()

        self._use_self_attn = use_self_attn
        self._use_cross_attn = use_cross_attn

        self.phi = MLP(input_dim=x_dim+y_dim,
                            output_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_phi,
                            use_bias=use_bias,
                            hidden_activation=nn.GELU())

        self.rho = MLP(input_dim=hidden_dim,
                            output_dim=determ_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_rho,
                            use_bias=use_bias,
                            hidden_activation=nn.GELU())
        
        if use_cross_attn:
            self.cross_attention_block = CrossAttention(query_dim=hidden_dim,
                                                        key_dim=hidden_dim,
                                                        value_dim=hidden_dim,
                                                        embed_dim=hidden_dim,
                                                        output_dim=determ_dim,
                                                        num_heads=8)
            self.x_to_querykey = MLP(input_dim=x_dim,
                                          output_dim=hidden_dim,
                                          hidden_dim=hidden_dim,
                                          n_h_layers=2,
                                          use_bias=use_bias,
                                          hidden_activation=nn.GELU())
        if use_self_attn:
            pass
        
    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                x_target: torch.Tensor | None = None,
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
            Shape (batch_size, determ_dim)
        """
        
        xy_input = torch.cat((x_context, y_context), dim=2)

        xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points[context], hidden_dim)

        if self._use_self_attn:
            pass

        if self._use_cross_attn:
            query = self.x_to_querykey(x_target) # Shape (batch_size, num_target_points, hidden_dim)
            key = self.x_to_querykey(x_context) # Shape (batch_size, num_context_points, hidden_dim)
            value = xy_encoded # Shape (batch_size, num_context_points, hidden_dim)

            determ_output = self.cross_attention_block(query, key, value) # Shape (batch_size, num_target_points, hidden_dim)
            assert determ_output.size(1) == x_target.size(1)
            return determ_output
            
        else:
            mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)
            determ_output = self.rho(mean_repr)
            assert determ_output.size(1) == 1

            return determ_output # Shape (batch_size, 1, determ_dim)

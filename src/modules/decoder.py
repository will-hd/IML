import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent
from .multivariate_diag import MultivariateNormalDiag
from .attention_modules import MultiheadAttention, CrossAttention
from typing import Callable
from .mlp import MLP

from typing import Literal

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 x_target_dim: int, 
                 y_dim: int, 
                 hidden_dim: int, 
                 latent_dim: int,
                 determ_dim: int,
                 n_h_layers: int,
                 path: Literal['latent', 'deterministic', 'both'],
                 use_bias: bool
                 ):

        super().__init__()
        
        self._path = path

        if path == 'both':
            decoder_input_dim = x_target_dim + latent_dim + determ_dim
        elif path == 'latent':
            decoder_input_dim = x_target_dim + latent_dim
        elif path == 'deterministic':
            decoder_input_dim = x_target_dim + determ_dim

        self.decoder = MLP(input_dim=decoder_input_dim,
                                output_dim=2*y_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers,
                                use_bias=use_bias,
                                hidden_activation=nn.GELU())

    def forward(self,
                x_target: torch.Tensor,
                z_samples: torch.Tensor,
                r: None | torch.Tensor = None
                ) -> Independent:
        """
        Parameters
        ----------
        x_target : torch.Tensor
            Shape (batch_size, num_target_points, x_target_dim)
        z_samples : torch.Tensor
            Shape (num_z_samples, batch_size, 1, latent_dim)
        r : torch.Tensor
            Shape (batch_size, 1, determ_dim) or (batch_size, num_target_points, determ_dim)
        Returns
        -------
        p_y_pred : torch.distributions.normal.Normal
            batch_shape (num_z_samples, batch_size, num_target_points)
            event_shape (latent_dim) 
        """

        _, num_target_points, _ = x_target.size()
        assert z_samples.size(-2) == 1
        z_samples = z_samples.expand(-1, -1, num_target_points, -1)

#        if self._use_deterministic_path:
#            assert r is not None, "use_deterministic_path is True but r is NOT PROVIDED"
#
#            if r.size(1) == 1:
#                r = r.repeat(1, num_target_points, 1)
#            elif r.size(1) != num_target_points:
#                raise ValueError("r must have size 1 or num_target_points in the second dimension")
#
#            decoder_input = torch.cat((x, z, r), dim=2)

        # assert r is None, "use_deterministic_path is False but r IS PROVIDED"
        x_target = x_target.unsqueeze(0).expand(z_samples.size(0), -1, -1, -1)
        decoder_input = torch.cat((x_target, z_samples), dim=-1)

        decoder_output = self.decoder(decoder_input) # Shape (num_z_samples, batch_size, num_target_points, 2*y_dim)

        mean, pre_stddev = torch.chunk(decoder_output, 2, dim=-1)
        stddev = 0.1 + 0.9 * F.softplus(pre_stddev)
        return MultivariateNormalDiag(mean, stddev) # Shape (num_z_samples, batch_size, num_target_points, y_dim)



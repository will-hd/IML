import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Callable

def make_MLP(
             input_dim: int,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

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



class LatentEncoder(nn.Module):
    """
    Latent encoder using DeepSets architecture (i.e. MLP for each input, sum, then MLP the sum to output)
    This consists of a 'phi' and 'rho' network: output = \\rho ( \\sum_i \\phi(x_i) )
    """
    def __init__(self, 
                 x_dim: int = 1,
                 y_dim: int = 1,
                 hidden_dim: int = 128,
                 latent_dim: int = 128, # i.e. z_dim
                 n_h_layers_phi: int = 3,
                 n_h_layers_rho: int = 2,
                 use_self_attn: bool = False,
                 use_knowledge: bool = False,
                 use_bias: bool = True,
#                 self_attention_type="dot",
#                 n_encoder_layers=3,
#                 min_std=0.01,
#                 batchnorm=False,
#                 dropout=0,
#                 attention_dropout=0,
#                 use_lvar=False,
#                 use_self_attn=False,
#                 attention_layers=2,
#                 use_lstm=False
                 ):

        super().__init__()

        self._use_self_attn = use_self_attn
        self._use_knowledge = use_knowledge

        self.phi = BatchMLP(input_dim=x_dim+y_dim,
                            output_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_phi,
                            use_bias=use_bias,
                            hidden_activation=nn.ReLU(),
                            output_activation=nn.Identity())
        
        self.rho = BatchMLP(input_dim=hidden_dim,
                            output_dim=2*latent_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_rho,
                            use_bias=use_bias,
                            hidden_activation=nn.ReLU(),
                            output_activation=nn.Identity())
        if use_self_attn:
            pass

        if use_knowledge:
            pass


        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                k: torch.Tensor | None = None) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        Returns
        -------
        q_Z : torch.distributions.normal.Normal
            Distribution q(z|x,y)
            Shape (batch_size, 1, latent_dim)
        """
    

        xy_input = torch.cat((x, y), dim=2)

        xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points, hidden_dim)
        
        if self._use_self_attn:
            pass
        else:
            mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)

        # Knowledge aggregation
        if k is not None:
            assert self._use_knowledge, "k is provided but use_knowledge is False"
        if self._use_knowledge:
            assert k is not None, "use_knowledge is True but k is None"
            # self.knowledge_aggregator(mean_repr, knowledge)
        
        latent_output = self.rho(mean_repr) # Shape (batch_size, 2*latent_dim)

        mean, pre_stddev = torch.chunk(latent_output, 2, dim=1)
        stddev = 0.1 + 0.9*F.sigmoid(pre_stddev) # Shape (batch_size, latent_dim)

        return Normal(mean, stddev) # Distribution q(z|x,y): Shape (batch_size, 1, latent_dim)



class DeterminisitcEncoder(nn.Module):
    """
    Deterministic encoder
    """
    def __init__(self,
                 x_dim: int = 1,
                 y_dim: int = 1,
                 hidden_dim: int = 128,
                 determ_dim: int = 128, # i.e. z_dim
                 n_h_layers_phi: int = 3,
                 n_h_layers_rho: int = 0,
                 use_cross_attn: bool = False,
                 use_self_attn: bool = False,
                 use_bias: bool = True
                 ):
        
        super().__init__()

        self._use_self_attn = use_self_attn
        self._use_cross_attn = use_cross_attn

        self.phi = BatchMLP(input_dim=x_dim+y_dim,
                            output_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_phi,
                            use_bias=use_bias,
                            hidden_activation=nn.ReLU(),
                            output_activation=nn.Identity())

        if n_h_layers_rho == 0:
            self.rho = nn.Linear(hidden_dim, determ_dim, bias=use_bias) # Equivalent to making ouput_dim=determ_dim in phi
        else:
            self.rho = BatchMLP(input_dim=hidden_dim,
                                output_dim=determ_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers_rho,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())
        
        if use_cross_attn:
            pass
        if use_self_attn:
            pass
        
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
            Shape (batch_size, determ_dim)
        """
        
        xy_input = torch.cat((x, y), dim=2)

        xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points, hidden_dim)

        if self._use_self_attn:
            pass

        if self._use_cross_attn:
            pass
        else:
            mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)
            determ_output = self.rho(mean_repr)

            return determ_output # Shape (batch_size, determ_dim)
            
        



class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 x_dim: int = 1, 
                 y_dim: int = 1, 
                 hidden_dim: int = 128, 
                 latent_dim: int = 128,
                 determ_dim: int = 128,
                 n_h_layers: int = 4,
                 use_deterministic_path: bool = False,
                 use_bias: bool = True
                 ):

        super().__init__()
        
        self._use_deterministic_path = use_deterministic_path
        self.target_transform = nn.Linear(x_dim, hidden_dim)

        if use_deterministic_path:
            decoder_input_dim = hidden_dim + latent_dim + determ_dim
        else:
            decoder_input_dim = hidden_dim + latent_dim

        self.decoder = BatchMLP(input_dim=decoder_input_dim,
                                output_dim=2*y_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())


    def forward(self,
                x_target: torch.Tensor,
                z: torch.Tensor,
                r: None | torch.Tensor = None
                ) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x_target : torch.Tensor
            Shape (batch_size, num_target_points, x_dim)
        z : torch.Tensor
            Shape (batch_size, 1, latent_dim)
        r : torch.Tensor
            Shape (batch_size, 1, determ_dim) or (batch_size, num_target_points, determ_dim)
        Returns
        -------
        p_y_pred : torch.distributions.normal.Normal
            Shape (batch_size, num_target_points, y_dim)
        """
        x = self.target_transform(x_target) # Shape (batch_size, num_target_points, hidden_dim)
        _, num_target_points, _ = x.size()

        assert z.size(1) == 1
        z = z.repeat(1, num_target_points, 1)

        if self._use_deterministic_path:
            assert r is not None, "use_deterministic_path is True but r is NOT PROVIDED"

            if r.size(1) == 1:
                r = r.repeat(1, num_target_points, 1)
            elif r.size(1) != num_target_points:
                raise ValueError("r must have size 1 or num_target_points in the second dimension")

            decoder_input = torch.cat((x, z, r), dim=2)

        else:
            assert r is None, "use_deterministic_path is False but r IS PROVIDED"
            decoder_input = torch.cat((x, z), dim=2)
        
        decoder_output = self.decoder(decoder_input) # Shape (batch_size, num_target_points, 2*y_dim)

        mean, pre_stddev = torch.chunk(decoder_output, 2, dim=2)
        stddev = 0.1 + 0.9 * F.softplus(pre_stddev)

        return Normal(mean, stddev) # Shape (batch_size, num_target_points, y_dim)

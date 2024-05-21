import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Callable

def make_MLP(
             in_size: int,
             out_size: int,
             hidden_size: int,
             num_h_layers: int, # number of hidden layers
             bias: bool,
             hidden_acc: Callable,
             output_acc: Callable
             ) -> list[nn.Module]:
    h = [hidden_size] * (num_h_layers)

    layers = []
    for i, (n, m) in enumerate(zip([in_size] + h, h + [out_size])):
        layers.append(nn.Linear(n, m, bias=bias))
        if i != num_h_layers:
            layers.append(hidden_acc)
        else:
            layers.append(output_acc)

    return layers

class LatentEncoder(nn.Module):
    """
    Latent encoder using DeepSets architecture (i.e. MLP for each input, sum, then MLP the sum to output)
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h_size: int = 128,
                 z_size: int = 64,
                 N_xy_to_si_layers: int = 2,
                 N_sc_to_qz_layers: int = 1
                 ):

        super().__init__()

        xy_to_si_layers = make_MLP(in_size=x_size+y_size,
                                   out_size=h_size,
                                   hidden_size=h_size,
                                   num_h_layers=N_xy_to_si_layers,
                                   bias=True,
                                   hidden_acc=nn.ReLU(),
                                   output_acc=nn.Identity())

        sc_to_qz_layers = make_MLP(in_size=h_size,
                                   out_size=2*z_size,
                                   hidden_size=h_size,
                                   num_h_layers=N_sc_to_qz_layers,
                                   bias=True,
                                   hidden_acc=nn.ReLU(),
                                   output_acc=nn.Identity())


        self.xy_to_si = nn.Sequential(*xy_to_si_layers)
        self.sc_to_qz = nn.Sequential(*sc_to_qz_layers)
    
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_size)
        y : torch.Tensor
            Shape (batch_size, num_points, y_size)
        Returns
        -------
        q_Z : torch.distributions.normal.Normal
            Shape (batch_size, z_size)
        """

        xy = torch.cat((x, y), dim=2)
        sc = self.xy_to_si(xy).mean(dim=1) # Shape (batch_size, h_size)
        
        qz_params = self.sc_to_qz(sc) # Shape (batch_size, 2*z_size)

        zloc, pre_zscale = torch.chunk(qz_params, 2, dim=1)
        zscale = 0.1 + 0.9*F.sigmoid(pre_zscale) # Shape (batch_size, z_size)

        return Normal(zloc, zscale) # Shape (batch_size, z_size)



class DeterminisitcEncoder(nn.Module):
    """
    Deterministic encoder
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h_size: int = 128,
                 r_size: int = 64,
                 N_h_layers: int = 6
                 ):
        
        super().__init__()
        xy_to_ri_layers = make_MLP(in_size=x_size+y_size,
                                   out_size=r_size,
                                   hidden_size=h_size,
                                   num_h_layers=N_h_layers,
                                   bias=True,
                                   hidden_acc=nn.ReLU(),
                                   output_acc=nn.Identity())

        self.deterministic_encoder = nn.Sequential(*xy_to_ri_layers) 
        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_size)
        y : torch.Tensor
            Shape (batch_size, num_points, y_size)
        Returns
        -------
        r : torch.Tensor
            Shape (batch_size, r_size)
        """
        
        xy = torch.cat((x, y), dim=2)
        ri = self.deterministic_encoder(xy) # Shape (batch_size, num_points, r_size)
        rc = ri.mean(dim=1) # Shape (batch_size, r_size)
        return rc

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h_size: int = 128, 
                 r_size: int = 64,
                 z_size: int = 64,
                 N_h_layers: int = 3,
                 use_r: bool = False
                 ):

        super().__init__()
        
        self.use_r = use_r
        if use_r:
            xzr_to_py_layers = make_MLP(in_size=x_size+r_size+z_size,
                                       out_size=2*y_size,
                                       hidden_size=h_size,
                                       num_h_layers=N_h_layers,
                                       bias=True,
                                       hidden_acc=nn.ReLU(),
                                       output_acc=nn.Identity())
            self.xzr_to_py = nn.Sequential(*xzr_to_py_layers)

        else:
            xz_to_py_layers = make_MLP(in_size=x_size+z_size,
                                       out_size=2*y_size,
                                       hidden_size=h_size,
                                       num_h_layers=N_h_layers,
                                       bias=True,
                                       hidden_acc=nn.ReLU(),
                                       output_acc=nn.Identity())
            self.xz_to_py = nn.Sequential(*xz_to_py_layers)


    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor,
                r: None | torch.Tensor = None
                ) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_target_points, x_size)
        z : torch.Tensor
            Shape (batch_size, z_size)
        r : torch.Tensor
            Shape (batch_size, r_size)
        Returns
        -------
        p_y_pred : torch.distributions.normal.Normal
            Shape (batch_size, num_target_points, y_size)
        """
        _, num_target_points, _ = x.size()
        z = z.unsqueeze(1).repeat(1, num_target_points, 1)

        if r is not None:
            assert self.use_r
            r = r.unsqueeze(1).repeat(1, num_target_points, 1)
            xzr = torch.cat((x, r, z), dim=2) # Shape (batch_size, num_target_points, x_+r_+z_size)
            py_params = self.xzr_to_py(xzr)

        else:
            xz = torch.cat((x, z), dim=2) # Shape (batch_size, num_target_points, x_+z_size)
            py_params = self.xz_to_py(xz)

        yloc, pre_yscale = torch.chunk(py_params, 2, dim=2)
        yscale = 0.1 + 0.9 * F.softplus(pre_yscale)

        return Normal(yloc, yscale)

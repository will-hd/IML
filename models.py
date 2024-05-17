import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class LatentEncoder(nn.Module):
    """
    Latent encoder
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h1_size: int = 128, 
                 h2_size: int = 96, 
                 z_size: int = 64
                 ):

        super().__init__()

        XY_to_h1_layers = [nn.Linear(x_size + y_size, h1_size), #1
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #2
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #3
                  nn.Identity()]
        self.XY_to_h1 = nn.Sequential(*XY_to_h1_layers)

        h1_to_h2_layers = [nn.Linear(h1_size, h2_size),
                           nn.ReLU()]
        self.h1_to_h2 = nn.Sequential(*h1_to_h2_layers)

        h2_to_loc_layers = [nn.Linear(h2_size, z_size),
                            nn.Identity()]
        self.h2_to_zloc = nn.Sequential(*h2_to_loc_layers)

        h2_to_scale_layers = [nn.Linear(h2_size, z_size),
                              nn.Identity()]
        self.h2_to_zscale = nn.Sequential(*h2_to_scale_layers)
    
        
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
        h1 = self.XY_to_h1(xy).mean(dim=1) # Shape (batch_size, h1_size)
        h2 = self.h1_to_h2(h1)

        zloc = self.h2_to_zloc(h2) # Shape (batch_size, z_size)
        zscale = 0.1 + 0.9*F.sigmoid(self.h2_to_zscale(h2)) # Shape (batch_size, z_size)

        q_Z = Normal(zloc, zscale) # Shape (batch_size, z_size)
        
        return q_Z


class DeterminisitcEncoder(nn.Module):
    """
    Deterministic encoder
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h1_size: int = 128, 
                 h2_size: int = 64, 
                 r_size: int = 64
                 ):
        
        super().__init__()
        XY_to_h1_layers = [nn.Linear(x_size + y_size, h1_size), #1
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #2
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #3
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #4
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #5
                  nn.ReLU(),
                  nn.Linear(h1_size, h1_size), #6
                  nn.ReLU()]

        h1_to_h2_layers = [nn.Linear(h1_size, h2_size),
                    nn.ReLU(),
                    nn.Linear(h2_size, r_size),
                    nn.Identity()]

        layers = [*XY_to_h1_layers, *h1_to_h2_layers]
        
        self.deterministic_encoder = nn.Sequential(*layers) 
        
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
        r_i = self.deterministic_encoder(xy) # Shape (batch_size, num_points, r_size)
        r = r_i.mean(dim=1) # Shape (batch_size, r_size)
        return r

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h_size: int = 128, 
                 r_size: int = 64,
                 z_size: int = 64
                 ):

        super().__init__()

        xrz_to_h_layers = [nn.Linear(x_size+z_size, h_size),
                  nn.ReLU(),
                  nn.Linear(h_size, h_size),
                  nn.ReLU()]
        h_to_yloc_layers = [nn.Linear(h_size, y_size),
                     nn.Identity()]
        h_to_yscale_layers = [nn.Linear(h_size, y_size),
                     nn.Identity()]

        self.xrz_to_h = nn.Sequential(*xrz_to_h_layers)
        self.h_to_yloc = nn.Sequential(*h_to_yloc_layers)
        self.h_to_yscale = nn.Sequential(*h_to_yscale_layers)

    def forward(self,
                x: torch.Tensor,
                r: torch.Tensor,
                z: torch.Tensor
                ) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_target_points, x_size)
        r : torch.Tensor
            Shape (batch_size, r_size)
        z : torch.Tensor
            Shape (batch_size, z_size)
        Returns
        -------
        p_y_pred : torch.distributions.normal.Normal
            Shape (batch_size, num_target_points, y_size)
        """
        _, num_target_points, _ = x.size()
        z = z.unsqueeze(1).repeat(1, num_target_points, 1)
        # r = r.unsqueeze(1).repeat(1, num_target_points, 1)
        xrz = torch.cat((x, z), dim=2)

        h = self.xrz_to_h(xrz)

        yloc = self.h_to_yloc(h)
        yscale = 0.1 + 0.9 * F.softplus(self.h_to_yloc(h))
        return Normal(yloc, yscale)

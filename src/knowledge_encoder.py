import torch
import torch.nn as nn

from .utils import make_MLP


class KnowledgeEncoder(nn.Module):
    """
    Latent encoder using DeepSets architecture (i.e. MLP for each input, sum, then MLP the sum to output)
    """
    def __init__(self, 
                 xK_size: int = 1, 
                 h_size: int = 128,
                 k_size: int = 128,
                 N_h_layer_phi: int = 2,
                 N_h_layer_rho: int = 2,
                 ):

        super().__init__()

        xK_to_phi_layers = make_MLP(in_size=xK_size,
                                   out_size=h_size,
                                   hidden_size=h_size,
                                   num_h_layers=N_h_layer_phi,
                                   bias=True,
                                   hidden_acc=nn.ReLU(),
                                   output_acc=nn.Identity())

        phi_to_rho_layers = make_MLP(in_size=h_size,
                                   out_size=k_size,
                                   hidden_size=h_size,
                                   num_h_layers=N_h_layer_rho,
                                   bias=True,
                                   hidden_acc=nn.ReLU(),
                                   output_acc=nn.Identity())


        self.phi = nn.Sequential(*xK_to_phi_layers)
        self.rho = nn.Sequential(*phi_to_rho_layers)
    
        
    def forward(self, xK: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xK : torch.Tensor
            Shape (batch_size, num_points, xK_size)
        Returns
        -------
        k : torch.Tensor 
            Shape (batch_size, k_size)
        """

        phi_i = self.phi(xK) # Shape (batch_size, num_points, h_size)

        phi = phi_i.mean(dim=1) # Shape (batch_size, h_size)

        k = self.rho(phi) # Shape (batch_size, k_size)

        return k # Shape (batch_size, k_size)


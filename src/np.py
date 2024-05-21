import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .mlp import MLP
from .models import LatentEncoder, DeterminisitcEncoder, Decoder


class NeuralProcess(nn.Module):

    def __init__(self,
                 x_size: int = 2,
                 y_size: int = 1,
                 r_size: int = 64,
                 z_size: int = 64,
                 h_size_dec: int = 128,
                 h_size_enc_lat: int = 128,
                 h_size_enc_det: int = 128,
                 N_h_layers_dec: int = 3,
                 N_h_layers_enc_lat_phi: int = 2,
                 N_h_layers_enc_lat_rho: int = 1,
                 #N_xy_to_si_layers: int = 2,
                 #N_sc_to_qz_layers: int = 1,
                 N_h_layers_enc_det: int = 6,
                 use_r: bool | None = None
                 ) -> None:
        super().__init__() 
        
        self.use_r = use_r
        assert use_r is not None, "use_r must be specified"

        self.latent_encoder = LatentEncoder(x_size, y_size, h_size_enc_lat,
                                            z_size, N_h_layers_enc_lat_phi, N_h_layers_enc_lat_rho)
        if use_r:
            self.deterministic_encoder = DeterminisitcEncoder(x_size, y_size, 
                                                              h_size_enc_det, r_size,
                                                              N_h_layers_enc_det)
        self.decoder = Decoder(x_size, y_size, h_size_dec, r_size,
                               z_size, N_h_layers_dec, use_r)

    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                x_target: torch.Tensor,
                y_target: None | torch.Tensor = None,
                ):

        if self.use_r:
            r_context = self.deterministic_encoder(x_context, y_context)
        else:
            r_context = None

        if self.training:
            # assumes that x_target includes the x_context points

            q_z_context = self.latent_encoder(x_context, y_context)
            q_z_target = self.latent_encoder(x_target, y_target)
            
            z_target_sample = q_z_target.rsample()
            p_y_pred = self.decoder(x=x_target, z=z_target_sample, r=r_context)

            return p_y_pred, q_z_target, q_z_context
        else:

            q_z_context = self.latent_encoder(x_context, y_context)
            z_context_sample = q_z_context.rsample()

            p_y_pred = self.decoder(x=x_target, z=z_context_sample, r=r_context) 

            return p_y_pred


if __name__ == "__main__":
    pass
    


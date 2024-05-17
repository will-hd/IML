import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from mlp import MLP
from models import LatentEncoder, DeterminisitcEncoder, Decoder


class NeuralProcess(nn.Module):

    def __init__(self,
                 x_size: int = 1,
                 y_size: int = 1,
                 r_size: int = 64,
                 z_size: int = 64,
                 h_size_dec: int = 128,
                 h1_size_lat: int = 128,
                 h2_size_lat: int = 96,
                 h1_size_det: int = 128,
                 h2_size_det: int = 64
                 ) -> None:
        super().__init__() 

        self.latent_encoder = LatentEncoder(x_size, y_size, h1_size_lat, h2_size_lat, z_size)
        self.deterministic_encoder = DeterminisitcEncoder(x_size, y_size, h1_size_det, h2_size_det, r_size)
        self.decoder = Decoder(x_size, y_size, h_size_dec, r_size, z_size)

    def forward(self,
                x_cntxt: torch.Tensor,
                y_cntxt: torch.Tensor,
                x_trgt: torch.Tensor,
                y_trgt: None | torch.Tensor = None
                ):

        if self.training:
            # assumes that x_trgt includes the x_context points

            q_z_cntxt = self.latent_encoder(x_cntxt, y_cntxt)
            q_z_trgt = self.latent_encoder(x_trgt, y_trgt)
            r_cntxt = self.deterministic_encoder(x_cntxt, y_cntxt)
            
            z_trgt_sample = q_z_trgt.rsample()
            p_y_pred = self.decoder(x_trgt, r_cntxt, z_trgt_sample)

            return p_y_pred, q_z_trgt, q_z_cntxt
        else:

            q_z_cntxt = self.latent_encoder(x_cntxt, y_cntxt)
            z_cntxt_sample = q_z_cntxt.rsample()
            r_cntxt = self.deterministic_encoder(x_cntxt, y_cntxt)

            p_y_pred = self.decoder(x_trgt, r_cntxt, z_cntxt_sample) 

            return p_y_pred


if __name__ == "__main__":
    pass
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .mlp import MLP
from .models import LatentEncoder, DeterminisitcEncoder, Decoder
from .knowledge_encoder import KnowledgeEncoder
from .models import make_MLP

import logging
logger = logging.getLogger(__name__)

class InformedNeuralProcess(nn.Module):

    def __init__(self,
                 x_size: int = 1,
                 y_size: int = 1,
                 r_size: int = 64,
                 z_size: int = 64,
                 xK_size: int = 3,
                 k_size: int = 128,
                 h_size_dec: int = 128,
                 h_size_enc_det: int = 128,
                 h_size_enc_know: int = 128,
                 h_size_agg: int = 128,
                 N_h_layers_dec: int = 3,
                 N_h_layers_enc_know_phi: int = 2,
                 N_h_layers_enc_know_rho: int = 2,
                 N_h_layers_enc_det: int = 6,
                 N_h_layers_agg: int = 3,
                 ) -> None:
        super().__init__() 
        

        self.deterministic_encoder = DeterminisitcEncoder(x_size, y_size, 
                                                          h_size_enc_det, r_size,
                                                              N_h_layers_enc_det)
        self.knowledge_encoder = KnowledgeEncoder(xK_size, h_size_enc_know, k_size, N_h_layers_enc_know_phi, 
                                                  N_h_layers_enc_know_rho) 
        self.decoder = Decoder(x_size, y_size, h_size_dec, r_size,
                               z_size, N_h_layers_dec, use_r=False)

        assert k_size == r_size, "k_size must equal r_size when using Sum+MLP agg"
        agg_layers = make_MLP(in_size=k_size,
                              out_size=2*z_size,
                              hidden_size=h_size_agg,
                              num_h_layers=N_h_layers_agg,
                              bias=True,
                              hidden_acc=nn.ReLU(),
                              output_acc=nn.Identity()
                              )
                              
        self.agg_mlp = nn.Sequential(*agg_layers)
    
    def aggregate(self,
            r_context: torch.Tensor, 
            k: torch.Tensor | None = None,
            ):

        if k is None: # k is none to make a normal NP
            a = self.agg_mlp(r_context)

        else:
            assert r_context.shape == k.shape # Shape (batch_size, k_size)
            a = self.agg_mlp(r_context+k)

        return a

    def latent_encoder(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       knowledge: torch.Tensor | None = None,
                       ) -> torch.distributions.normal.Normal:

        r = self.deterministic_encoder(x, y)
        if knowledge is not None:
            k = self.knowledge_encoder(knowledge)
        else:
            k = None
        a = self.aggregate(r, k)

        zloc, pre_zscale = torch.chunk(a, 2, dim=1)
        zscale = 0.1 + 0.9*F.sigmoid(pre_zscale) # Shape (batch_size, z_size)

        q_z = Normal(zloc, zscale) 

        return q_z

    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                knowledge: torch.Tensor,
                x_target: torch.Tensor,
                y_target: None | torch.Tensor = None,
                ):



        if self.training:
            # assumes that x_target includes the x_context points

            q_z_context = self.latent_encoder(x_context, y_context, knowledge)
            q_z_target = self.latent_encoder(x_target, y_target, knowledge)
            
            z_target_sample = q_z_target.rsample()
            p_y_pred = self.decoder(x=x_target, z=z_target_sample, r=None)

            return p_y_pred, q_z_target, q_z_context
        else:

            q_z_context = self.latent_encoder(x_context, y_context, knowledge)
            z_context_sample = q_z_context.rsample()

            p_y_pred = self.decoder(x=x_target, z=z_context_sample, r=None) 

            return p_y_pred

    @property
    def device(self):
        return next(self.parameters()).device

if __name__ == "__main__":
    pass
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .mlp import MLP
from .models import LatentEncoder, DeterminisitcEncoder, Decoder, KnowledgeDecoder

class LUPINeuralProcess(nn.Module):

    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 knowledge_dim: int,
                 hidden_dim: int = 128,
                 latent_dim: int = 128,
                 determ_dim: int = 128,
                 n_h_layers_decoder: int = 4,
                 n_h_layers_knowledge_decoder: int = 4,
                 n_h_layers_phi_latent_encoder: int = 3,
                 n_h_layers_rho_latent_encoder: int = 2,
                 n_h_layers_phi_determ_encoder: int = 3,
                 n_h_layers_rho_determ_encoder: int = 0,
                 use_deterministic_path: bool = False,
                 use_bias: bool = True,
                 user_context_in_target: bool = True # TODO investigate
                 ) -> None:
        super().__init__() 

        self._use_deterministic_path = use_deterministic_path

        if use_deterministic_path:
            self.deterministic_encoder = DeterminisitcEncoder(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    determ_dim=determ_dim,
                    n_h_layers_phi=n_h_layers_phi_determ_encoder,
                    n_h_layers_rho=n_h_layers_rho_determ_encoder,
                    use_cross_attn=False,
                    use_self_attn=False,
                    use_bias = use_bias
            )

        self.latent_encoder = LatentEncoder(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    n_h_layers_phi=n_h_layers_phi_latent_encoder,
                    n_h_layers_rho=n_h_layers_rho_latent_encoder,
                    use_self_attn=False,
                    use_knowledge=False,
                    use_bias=use_bias,
                    return_mean_repr=True
        )

        self.decoder = Decoder(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    determ_dim=determ_dim,
                    n_h_layers=n_h_layers_decoder,
                    use_bias=use_bias
        )

        self.knowledge_decoder = KnowledgeDecoder(
                    knowledge_dim=knowledge_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    determ_dim=determ_dim,
                    n_h_layers=n_h_layers_knowledge_decoder,
                    use_bias=use_bias
        )
                    

    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                x_target: torch.Tensor,
                knowledge: torch.Tensor,
                y_target: None | torch.Tensor = None,
                ):

        if self._use_deterministic_path:
            r_context = self.deterministic_encoder(x_context, y_context)
        else:
            r_context = None

        z_prior_dist, encoded_context = self.latent_encoder(x_context, y_context) # Shape (batch_size, 1, latent_dim)

        if self.training:
            assert y_target is not None, "y_target must be provided during training"
            z_post_dist, _ = self.latent_encoder(x_target, y_target) # assumes that x_target includes the x_context points
            z = z_post_dist.rsample()

            p_y_pred = self.decoder(x_target=x_target,z=z,r=r_context)
            knowledge_pred = self.knowledge_decoder(encoded_context) # Shape (batch_size, 1, knowledge_dim)

            loss, log_p = self.calculate_loss(pred_dist=p_y_pred,
                                              y_target=y_target,
                                              posterior=z_post_dist,
                                              prior=z_prior_dist)

            if knowledge is not None:
                knowledge_loss = F.mse_loss(knowledge_pred, knowledge) 
        
                return p_y_pred, loss, knowledge_loss
            else:
                return p_y_pred, loss

            #loss = self._loss(p_y_pred, y_target, z_post_dist, z_prior_dist)


        else: 
            z = z_prior_dist.rsample()

            p_y_pred = self.decoder(x_target=x_target,z=z,r=r_context)

            return p_y_pred

    
    def calculate_loss(self,
                      pred_dist: Normal, 
                      y_target: torch.Tensor,
                      posterior: Normal,
                      prior: Normal
                      ):

        batch_size, num_targets, _ = y_target.shape
        log_p = pred_dist.log_prob(y_target).sum(-1) # Shape (batch_size, num_targets)
        # assert log_p.shape[-1] == 1
        # log_p = log_p.squeeze(-1)

        kl_div = torch.sum(kl_divergence(posterior, prior), dim=-1, keepdim=True) # Shape (batch_size, 1)

        loss = -torch.mean(log_p - kl_div / num_targets)
        return loss, log_p
    
    def _loss(self, p_y_pred, y_target, q_target, q_context):
            """
            Computes Neural Process loss.

            Parameters
            ----------
            p_y_pred : one of torch.distributions.Distribution
                Distribution over y output by Neural Process.

            y_target : torch.Tensor
                Shape (batch_size, num_target, y_dim)

            q_target : one of torch.distributions.Distribution
                Latent distribution for target points.

            q_context : one of torch.distributions.Distribution
                Latent distribution for context points.
            """
            # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
            # over batch and sum over number of targets and dimensions of y
            log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
            # KL has shape (batch_size, r_dim). Take mean over batch and sum over
            # r_dim (since r_dim is dimension of normal distribution)
            kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
            return -log_likelihood + kl

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    pass
    


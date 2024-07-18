import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .mlp import MLP
from .models import LatentEncoder, DeterminisitcEncoder, Decoder, KnowledgeEncoder

class NeuralProcess(nn.Module):

    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dim: int = 128,
                 latent_dim: int = 128,
                 determ_dim: int = 128,
                 knowledge_dim: int = 4,
                 n_h_layers_decoder: int = 4,
                 n_h_layers_phi_latent_encoder: int = 3,
                 n_h_layers_rho_latent_encoder: int = 2,
                 n_h_layers_phi_determ_encoder: int = 3,
                 n_h_layers_rho_determ_encoder: int = 0,
                 use_deterministic_path: bool = False,
                 use_bias: bool = True,
                 user_context_in_target: bool = True, # TODO investigate
                 use_knowledge: bool = False,
                 use_linear_knowledge_encoder=False,
                 use_latent_self_attn: bool = False,
                 use_determ_self_attn: bool = False,
                 use_determ_cross_attn: bool = False
                 ) -> None:
        super().__init__() 

        self._use_deterministic_path = use_deterministic_path
        self._use_determ_cross_attn = use_determ_cross_attn
        assert not (use_determ_cross_attn and not use_deterministic_path), "use_determ_cross_attn requires use_deterministic_path to be True"
        self._use_knowledge = use_knowledge
        self._knowledge_dim = knowledge_dim

        if use_deterministic_path:
            self.deterministic_encoder = DeterminisitcEncoder(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    determ_dim=determ_dim,
                    n_h_layers_phi=n_h_layers_phi_determ_encoder,
                    n_h_layers_rho=n_h_layers_rho_determ_encoder,
                    use_cross_attn=use_determ_cross_attn,
                    use_self_attn=use_determ_self_attn,
                    use_bias = use_bias
            )
        
        if use_knowledge:
            self.knowledge_encoder = KnowledgeEncoder(
                    knowledge_dim=knowledge_dim,
                    hidden_dim=hidden_dim,
                    n_h_layers_phi=2,
                    n_h_layers_rho=2,
                    use_bias=use_bias,
                    only_use_linear=use_linear_knowledge_encoder
            )


        self.latent_encoder = LatentEncoder(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    n_h_layers_phi=n_h_layers_phi_latent_encoder,
                    n_h_layers_rho=n_h_layers_rho_latent_encoder,
                    use_self_attn=use_latent_self_attn,
                    use_knowledge=use_knowledge,
                    use_bias=use_bias
        )

        self.decoder = Decoder(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    determ_dim=determ_dim,
                    n_h_layers=n_h_layers_decoder,
                    use_deterministic_path=use_deterministic_path,
                    use_bias=use_bias
        )

        # print("Neural Process model created")
        # for modules in self.modules():
        #     print(f"Module: {modules}")

        # for name, param in self.named_parameters():
        #     print(name, param.size())

    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                x_target: torch.Tensor,
                knowledge: torch.Tensor,
                y_target: None | torch.Tensor = None,
                ):

        if self._use_deterministic_path:
            if self._use_determ_cross_attn:
                r_context = self.deterministic_encoder(x_context, y_context, x_target) # Shape (batch_size, num_target_points, hidden_dim)
            else:
                r_context = self.deterministic_encoder(x_context, y_context) # Shape (batch_size, 1, hidden_dim)
        else:
            r_context = None

        if self._use_knowledge and knowledge is not None:
            #assert knowledge.shape[-1] == self._knowledge_dim, "knowledge does not have the correct shape"
            knowledge = self.knowledge_encoder(knowledge)
        else:
            knowledge = None

        z_prior_dist = self.latent_encoder(x_context, y_context, knowledge) # Shape (batch_size, 1, latent_dim)

        if self.training:
            assert y_target is not None, "y_target must be provided during training"
            z_post_dist = self.latent_encoder(x_target, y_target, knowledge) # assumes that x_target includes the x_context points
            z = z_post_dist.rsample()

            p_y_pred = self.decoder(x_target=x_target,z=z,r=r_context)
            
            loss, log_lik = self.calculate_loss(pred_dist=p_y_pred,
                                              y_target=y_target,
                                              posterior=z_post_dist,
                                              prior=z_prior_dist)
            #loss = self._loss(p_y_pred, y_target, z_post_dist, z_prior_dist)

            return p_y_pred, loss, log_lik

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
        log_lik = pred_dist.log_prob(y_target).sum(-1) # Shape (batch_size, num_targets)
        # assert log_p.shape[-1] == 1
        # log_p = log_p.squeeze(-1)

        kl_div = torch.sum(kl_divergence(posterior, prior), dim=-1, keepdim=True) # Shape (batch_size, 1)

        loss = -torch.mean(log_lik - kl_div / num_targets)
        return loss, log_lik
    
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
    
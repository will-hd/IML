import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.kl import kl_divergence
import math

from src.modules.mlp import MLP
from src.modules.xy_encoders import XEncoder, XYSetEncoder
from src.modules.latent_encoder import FiLMLatentEncoder
from src.modules.decoder import Decoder
from src.modules.deterministic_encoder import DeterminisitcEncoder
from src.modules.knowledge_encoder import RoBERTaKnowledgeEncoder

import logging
logger = logging.getLogger(__name__)

from typing import Literal

def sum_log_prob(dist, target):
    # assert len(dist.batch_shape) == 3
    
    return dist.log_prob(target).sum(dim=(-1)) 

class InformedNeuralProcess(nn.Module):

    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dim: int = 128,  # Dimension of all MLP hidden layers 
                 latent_dim: int = 128,  # Dimension of latent vector z
                 determ_dim: int = 128,
                 knowledge_dim: int = 128,  # Dimension of knowledge vector repr
                 mlps_activation: nn.Module = nn.GELU(), 
                 x_proj_dim: int = 128,
                 n_h_layers_x_proj: int = 1,
                 n_h_layers_decoder: int = 4,
                 n_h_layers_latent_xy_encoder: int = 3,
                 n_h_layers_film_latent_encoder: int = 2,
#                 n_h_layers_rho_latent_encoder: int = 0,
                 n_h_layers_phi_determ_encoder: int = 3,
                 n_h_layers_rho_determ_encoder: int = 0,
                 path: Literal['latent', 'deterministic', 'both'] = 'latent',
                 train_num_z_samples: int = 1,
                 test_num_z_samples: int = 32,
                 use_bias: bool = True,
                 user_context_in_target: bool = True, # TODO investigate
                 use_latent_self_attn: bool = False,
                 use_determ_self_attn: bool = False,
                 use_determ_cross_attn: bool = False,
                 use_knowledge: bool = True,
                 knowledge_dropout: float = 0.3,
                 roberta_return_cls: bool = True,
                 tune_llm_layer_norms: bool = True,
                 freeze_llm: bool = True,
                 knowledge_projection_n_h_layers: int = 0,
                 knowledge_aggregation_method: Literal['sum+MLP', 'FiLM+MLP'] = 'FiLM+MLP',
                 device: Literal['cuda', 'cpu'] = 'cuda',
                 beta: float = 1.0
                 ) -> None:
        super().__init__() 

        self._path = path
        self._use_determ_cross_attn = use_determ_cross_attn
        assert not (use_determ_cross_attn and (path == 'latent')), 'use_determ_cross_attn requires path to be deterministic or both'
        self._use_knowledge = use_knowledge
        self._knowledge_dim = knowledge_dim
        self._knowledge_dropout = knowledge_dropout

        self._train_num_z_samples = 1
        self._test_num_z_samples = 32

        self.beta = beta

        self.x_encoder = XEncoder(x_dim=x_dim,
                                  x_proj_dim=x_proj_dim,
                                  hidden_dim=hidden_dim,
                                  n_h_layers=n_h_layers_x_proj,
                                  activation=mlps_activation,
                                  use_bias=use_bias
                                  )

        if path in ['latent', 'both']:
            self.xy_encoder_latent = XYSetEncoder(
                    x_dim=x_proj_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    n_h_layers_phi=n_h_layers_latent_xy_encoder,
                    use_self_attn=use_latent_self_attn,
                    n_self_attn_heads=4,
                    set_agg_function='mean',
                    activation=mlps_activation,
                    use_bias=use_bias)

            self.latent_encoder = FiLMLatentEncoder(
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    knowledge_dim=knowledge_dim,
                    n_h_layers=n_h_layers_film_latent_encoder,
                    use_bias=use_bias,
                    activation=mlps_activation,
                    FiLM_before_activation=True)
        
        if use_knowledge:
            self.knowledge_encoder = RoBERTaKnowledgeEncoder(
                knowledge_dim=knowledge_dim,
                return_cls=roberta_return_cls,
                tune_llm_layer_norms=tune_llm_layer_norms,
                freeze_llm=freeze_llm,
                knowledge_projection_n_h_layers=knowledge_projection_n_h_layers,
                knowledge_projection_hidden_dim=hidden_dim,
                knowledge_projection_activation=mlps_activation,
                use_bias=use_bias,
                device=device,
            )


        if path in ['deterministic', 'both']:
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

        self.decoder = Decoder(
                    x_target_dim=x_proj_dim,
                    y_dim=y_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    determ_dim=determ_dim,
                    n_h_layers=n_h_layers_decoder,
                    path=path,
                    activation=mlps_activation,
                    use_bias=use_bias
        )


    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                x_target: torch.Tensor,
                textual_knowledge: tuple[str] | None = None,
                y_target: None | torch.Tensor = None,
                ) -> tuple[Independent, Independent, Independent | None]:

        x_context = self.x_encoder(x_context)
        x_target = self.x_encoder(x_target)

        # Knowledge to vector
        if (torch.rand(1) < self._knowledge_dropout and
                                        self.training) or textual_knowledge is None:
            k = None
        else:
            k = self.knowledge_encoder(textual_knowledge) # Shape (batch_size, 1, knowledge_dim)

        # Encode context set
        r_context = self.xy_encoder_latent(x_context, y_context) # Shape (batch_size, 1, hidden_dim)
        q_z_context = self.latent_encoder(r_context, k)


        if self.training:
            assert y_target is not None, "y_target must be provided during training"
            
            # Note: method assumes that x_target includes the x_context points
            r_target = self.xy_encoder_latent(x_target, y_target)  # Shape (batch_size, 1, hidden_dim)
            q_z_target = self.latent_encoder(r_target, k)  # batch_shape (batch_size, 1), event_shape (latent_dim) 

            z_samples = q_z_target.rsample([self._train_num_z_samples])  # Shape (num_z_samples, batch_size, 1, latent_dim)

            p_y_pred = self.decoder(x_target=x_target, z_samples=z_samples, r=None)  # batch_shape (num_z_samples, batch_size, num_target_points), event_shape (y_dim)
            
            return p_y_pred, q_z_context, q_z_target
        else: 
            z_samples = q_z_context.rsample([self._test_num_z_samples])  # Shape (num_z_samples, batch_size, 1, latent_dim)

            p_y_pred = self.decoder(x_target=x_target, z_samples=z_samples, r=None)

            return p_y_pred, q_z_context, None


#        if self._use_deterministic_path:
#            if self._use_determ_cross_attn:
#                r_context = self.deterministic_encoder(x_context, y_context, x_target) # Shape (batch_size, num_target_points, hidden_dim)
#            else:
#                r_context = self.deterministic_encoder(x_context, y_context) # Shape (batch_size, 1, hidden_dim)
#        else:
#            r_context = None
    
    @property
    def device(self):
        return next(self.parameters()).device

#---- Old Loss Functions ------
#    def calculate_loss(self,
#                      pred_dist: Normal, 
#                      y_target: torch.Tensor,
#                      posterior: Normal,
#                      prior: Normal
#                      ):
#
#        batch_size, num_targets, _ = y_target.shape
#        log_lik = pred_dist.log_prob(y_target).sum(-1) # Shape (batch_size, num_targets)
#        # assert log_p.shape[-1] == 1
#        # log_p = log_p.squeeze(-1)
#
#        kl_div = torch.sum(kl_divergence(posterior, prior), dim=-1, keepdim=True) # Shape (batch_size, 1)
#
#        loss = -torch.mean(log_lik - kl_div / num_targets)
#        return loss, log_lik
#    
#    def _loss(self, p_y_pred, y_target, q_target, q_context):
#            """
#            Computes Neural Process loss.
#
#            Parameters
#            ----------
#            p_y_pred : one of torch.distributions.Distribution
#                Distribution over y output by Neural Process.
#
#            y_target : torch.Tensor
#                Shape (batch_size, num_target, y_dim)
#
#            q_target : one of torch.distributions.Distribution
#                Latent distribution for target points.
#
#            q_context : one of torch.distributions.Distribution
#                Latent distribution for context points.
#            """
#            # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
#            # over batch and sum over number of targets and dimensions of y
#            log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
#            # KL has shape (batch_size, r_dim). Take mean over batch and sum over
#            # r_dim (since r_dim is dimension of normal distribution)
#            kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
#            return -log_likelihood + kl

if __name__ == "__main__":
    from utils import setup_logging
    setup_logging(console=True, file=False, debug=True)
    
    device = 'cuda'
    x_dim = 1
    y_dim = 1
    model = InformedNeuralProcess(x_dim=x_dim, y_dim=y_dim, device=device).to(device)

    BATCH_SIZE = 8
    NUM_CONTEXT_POINTS = 10
    NUM_TARGET_POINTS = 7
    x_context = torch.randn(BATCH_SIZE, NUM_CONTEXT_POINTS, 1).to(device)
    y_context = torch.randn(BATCH_SIZE, NUM_CONTEXT_POINTS, 1).to(device)
    x_target = torch.randn(BATCH_SIZE, NUM_TARGET_POINTS, 1).to(device)
    y_target = torch.randn(BATCH_SIZE, NUM_TARGET_POINTS, 1).to(device)

    textual_knowledge = ['hello world'] * BATCH_SIZE

    p_y_pred, q_z_context, q_z_target = model(x_context, y_context, x_target, textual_knowledge, y_target)


    logging.info('---------Testing----------')
    logging.info(f'p_y_pred.batch_shape: {p_y_pred.batch_shape}, p_y_pred.event_shape: {p_y_pred.event_shape}')
    logging.info(f'expect batch_shape: {model._train_num_z_samples, BATCH_SIZE, NUM_TARGET_POINTS}')
    logging.info(f'expect event_shape: {1}')
    logging.info(f'q_z_context.batch_shape: {q_z_context.batch_shape}, q_z_context.event_shape: {q_z_context.event_shape}')
    logging.info(f'Expect batch_shape: {BATCH_SIZE, 1}, event_shape: {128}')
    logging.info(f'q_z_target.batch_shape: {q_z_target.batch_shape}, q_z_target.event_shape: {q_z_target.event_shape}')
    logging.info(f'Expect batch_shape: {BATCH_SIZE, 1}, event_shape: {128}')
#    print(p_y_pred.batch_shape, p_y_pred.event_shape)
#    print(q_z_context.batch_shape, q_z_context.event_shape)
#    print(q_z_target.batch_shape, q_z_target.event_shape)
                    
                    


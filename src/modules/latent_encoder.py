import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent
# from .attention_modules import MultiheadAttention, CrossAttention
from .attention import MultiheadAttender
from typing import Callable
from .mlp import MLP
from .FiLM import LinearFiLMActBlock
from .multivariate_diag import MultivariateNormalDiag

from typing import Literal
import logging
logger = logging.getLogger(__name__)


class FiLMLatentEncoder(nn.Module):
    """
    Latent encoder using DeepSets architecture (i.e. MLP for each input, sum, then MLP the sum to output)
    This consists of a 'phi' and 'rho' network: output = \\rho ( \\sum_i \\phi(x_i) )
    """
    def __init__(self, 
                 hidden_dim: int,
                 latent_dim: int, # i.e. z_dim
                 knowledge_dim: int,
                 n_h_layers: int,
                 use_bias: bool,
                 FiLM_before_activation: bool,
                 ):

        super().__init__()

        self._n_h_layers = n_h_layers
        self.hidden_dim = hidden_dim

        self.FiLM_params_generator = nn.Linear(knowledge_dim, 2*hidden_dim*n_h_layers)
        
        assert n_h_layers > 0, "n_h_layers must be greater than 0 (NotImplemented yet....)"
        self.FiLM_blocks = nn.ModuleList(
                [LinearFiLMActBlock(
                                    hidden_dim, hidden_dim, 
                                    use_bias, nn.GELU(), FiLM_before_activation
                                    ) for _ in range(n_h_layers)]
            )
        
        self.output_layer = nn.Linear(hidden_dim, 2*latent_dim, bias=use_bias)

                

    def forward(self,
                r: torch.Tensor,
                k: torch.Tensor | None
                ) -> Independent:
        """
        Parameters
        ----------
        xy_encoding : torch.Tensor
            Dataset encoding (DeepSets rho+phi output)
            Shape (batch_size, 1, hidden_dim)
        knowledge_encoding : torch.Tensor
            Knowledge encoding
            Shape (batch_size, 1, knowledge_dim)
        Returns
        -------
        q_Z : torch.distributions.normal.Normal
            Distribution q(z|x,y)
            Batch Shape (batch_size, 1), Event Shape (latent_dim)
        """
        out = r
        if k is None:
            FiLM_params = [None]*self._n_h_layers
            for block, gamma_beta in zip(self.FiLM_blocks, FiLM_params):
                out = block(out, None, None)
        else:
            # Note: Kasia's code uses r and k in params generator....
            FiLM_params = self.FiLM_params_generator(k)  # Shape (batch_size, 1, 2*hidden_dim*n_h_layers)
            logging.debug(f'FiLM_params={FiLM_params.size()}')
            FiLM_params = torch.split(FiLM_params, 2*self.hidden_dim, dim=-1)
            logging.debug(f'len FiLM_params={len(FiLM_params)}')

            assert len(FiLM_params) == len(self.FiLM_blocks), "Number of FiLM params and FiLM blocks must be the same"
            for block, gamma_beta in zip(self.FiLM_blocks, FiLM_params):
                gamma, beta = torch.split(gamma_beta, self.hidden_dim, dim=-1) 
                logging.debug(f'gamma={gamma.size()} beta={beta.size()}')
                out = block(out, gamma, beta)
                logging.debug(f'out={out.size()}')
        
        latent_output = self.output_layer(out) # Shape (batch_size, 1, 2*latent_dim)
        logging.debug(f'latent_output={latent_output.size()}')
#        # Knowledge aggregation
#        if k is not None:
#            assert self._use_knowledge, "k is provided but use_knowledge is False"
#        if self._use_knowledge:
#            #assert k is not None, "use_knowledge is True but k is None"
#            latent_output = self.knowledge_aggregator(mean_repr, k) # Shape (batch_size, 2*latent_dim)
#        else:
#            latent_output = self.rho(mean_repr) # Shape (batch_size, 2*latent_dim)
            
        mean, pre_stddev = torch.chunk(latent_output, 2, dim=-1)
        stddev = 0.1 + 0.9*F.sigmoid(pre_stddev) # Shape (batch_size, 1, latent_dim)

        return MultivariateNormalDiag(mean, stddev) # q(z|x,y):  batch_shape (batch_size, 1), event_shape (latent_dim)


# class LatentEncoder(nn.Module):
#     """
#     Latent encoder using DeepSets architecture (i.e. MLP for each input, sum, then MLP the sum to output)
#     This consists of a 'phi' and 'rho' network: output = \\rho ( \\sum_i \\phi(x_i) )
#     """
#     def __init__(self, 
#                  x_dim: int,
#                  y_dim: int,
#                  hidden_dim: int,
#                  latent_dim: int, # i.e. z_dim
#                  n_h_layers_phi: int,
#                  n_h_layers_rho: int,
#                  use_self_attn: bool,
#                  use_knowledge: bool,
#                  use_bias: bool,
#                  return_mean_repr: bool,
#                  knowledge_aggregation_method: Literal['sum+MLP', 'FiLM+MLP'] | None
#                  ):
# 
#         super().__init__()
# 
#         self._use_self_attn = use_self_attn
#         self._use_knowledge = use_knowledge
#         self._return_mean_repr = return_mean_repr
#         self._knowledge_aggregation_method = knowledge_aggregation_method
# 
#         self.phi = MLP(input_dim=x_dim+y_dim,
#                             output_dim=hidden_dim,
#                             hidden_dim=hidden_dim,
#                             n_h_layers=n_h_layers_phi,
#                             use_bias=use_bias,
#                             hidden_activation=nn.GELU())
#         
#         if knowledge_aggregation_method == 'FiLM+MLP':
#             self.rho = FiLM_MLP(x_input_dim=hidden_dim,
#                                 k_input_dim=hidden_dim,
#                                 output_dim=2*latent_dim,
#                                 hidden_dim=hidden_dim,
#                                 n_h_layers=n_h_layers_rho,
#                                 use_bias=use_bias,
#                                 hidden_activation=nn.GELU(),
#                                 output_activation=nn.Identity())
#                 
#         else:
#             self.rho = MLP(input_dim=hidden_dim,
#                                 output_dim=2*latent_dim,
#                                 hidden_dim=hidden_dim,
#                                 n_h_layers=n_h_layers_rho,
#                                 use_bias=use_bias,
#                                 hidden_activation=nn.GELU())
# 
#         if use_self_attn:
#             self.self_attention_block = MultiheadAttention(input_dim=hidden_dim,
#                                                            embed_dim=hidden_dim,
#                                                            num_heads=8)
# 
#         if use_knowledge:
#             pass
# #            self.agg_mlp = MLP(input_dim=hidden_dim,
# #                                    ouput_dim=hidden_dim,
# #                                    hidden_dim=hidden_dim,
# #                                    n_h_layers=2,
# #                                    use_bias=use_bias,
# #                                    hidden_activation=nn.GELU(),
# #                                    output_activation=nn.Identity())
# 
#     def knowledge_aggregator(self,
#                              context_repr: torch.Tensor,
#                              knowledge_repr: torch.Tensor,
#                              ) -> torch.Tensor:
# 
#         if knowledge_repr is not None:
#             assert self._knowledge_aggregation_method in ['sum+MLP', 'FiLM+MLP', None], "Knowledge aggregation method must be one of ['sum+MLP', 'FiLM+MLP', None]"
#             if self._knowledge_aggregation_method == 'FiLM+MLP':
#                 return self.rho(x=context_repr, film_input=knowledge_repr)
#             elif self._knowledge_aggregation_method == 'sum+MLP':
#                 return self.rho(context_repr + knowledge_repr)
#         else:
#             assert self._use_knowledge == False, 'Must provided knowledge if using knowledge'
#             return context_repr
# 
#         
#     def forward(self,
#                 x: torch.Tensor,
#                 y: torch.Tensor,
#                 k: torch.Tensor | None = None) -> torch.distributions.normal.Normal | tuple[torch.distributions.normal.Normal, torch.Tensor]:
#         """
#         Parameters
#         ----------
#         x : torch.Tensor
#             Shape (batch_size, num_points, x_dim)
#         y : torch.Tensor
#             Shape (batch_size, num_points, y_dim)
#         Returns
#         -------
#         q_Z : torch.distributions.normal.Normal
#             Distribution q(z|x,y)
#             Shape (batch_size, 1, latent_dim)
#         """
#     
# 
#         xy_input = torch.cat((x, y), dim=2)
# 
#         xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points, hidden_dim)
#         
#         if self._use_self_attn:
#             xy_encoded = self.self_attention_block(xy_encoded) # Shape (batch_size, num_points, hidden_dim)
#         
#         mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)
# 
#         # Knowledge aggregation
#         if k is not None:
#             assert self._use_knowledge, "k is provided but use_knowledge is False"
#         if self._use_knowledge:
#             #assert k is not None, "use_knowledge is True but k is None"
#             latent_output = self.knowledge_aggregator(mean_repr, k) # Shape (batch_size, 2*latent_dim)
#         else:
#             latent_output = self.rho(mean_repr) # Shape (batch_size, 2*latent_dim)
#             
#         mean, pre_stddev = torch.chunk(latent_output, 2, dim=-1)
#         stddev = 0.1 + 0.9*F.sigmoid(pre_stddev) # Shape (batch_size, latent_dim)
# 
#         if self._return_mean_repr:
#             return MultivariateNormalDiag(mean, stddev), mean_repr
#         else:
#             return MultivariateNormalDiag(mean, stddev) # Distribution q(z|x,y): Shape (batch_size, 1, latent_dim)

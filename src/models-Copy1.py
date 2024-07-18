import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from .attention_modules import MultiheadAttention, CrossAttention
from typing import Callable

def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)

def make_MLP(
             input_dim: int,
             output_dim: int,
             hidden_dim: int,
             n_h_layers: int, # number of hidden layers
             use_bias: bool,
             hidden_activation: Callable,
             output_activation: Callable
             ) -> list[nn.Module]:
    h = [hidden_dim] * (n_h_layers)

    layers = []
    for i, (n, m) in enumerate(zip([input_dim] + h, h + [output_dim])):
        layers.append(nn.Linear(n, m, bias=use_bias))
        if i != n_h_layers:
            layers.append(hidden_activation)
        else:
            layers.append(output_activation)

    return layers

class BatchMLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_h_layers: int,
                 use_bias: bool,
                 hidden_activation: Callable,
                 output_activation: Callable
                 ) -> None:
        super().__init__()
        
        layers = self.make_MLP_layers(input_dim=input_dim,
                                      output_dim=output_dim,
                                      hidden_dim=hidden_dim,
                                      n_h_layers=n_h_layers,
                                      use_bias=use_bias,
                                      hidden_activation=hidden_activation,
                                      output_activation=output_activation)
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def _initialize_weights(self):
        for module in self.mlp:
            self._init_weights(module)


    @staticmethod
    def make_MLP_layers(input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_h_layers: int, # number of hidden layers
                 use_bias: bool,
                 hidden_activation: Callable,
                 output_activation: Callable
                 ) -> list[nn.Module]:

        h = [hidden_dim] * (n_h_layers)

        layers = []
        for i, (n, m) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(nn.Linear(n, m, bias=use_bias))
            if i != n_h_layers:
                layers.append(hidden_activation)
            else:
                layers.append(output_activation)

        return layers



class LatentEncoder(nn.Module):
    """
    Latent encoder using DeepSets architecture (i.e. MLP for each input, sum, then MLP the sum to output)
    This consists of a 'phi' and 'rho' network: output = \\rho ( \\sum_i \\phi(x_i) )
    """
    def __init__(self, 
                 x_dim: int,
                 y_dim: int,
                 hidden_dim: int = 128,
                 latent_dim: int = 128, # i.e. z_dim
                 n_h_layers_phi: int = 3,
                 n_h_layers_rho: int = 2,
                 use_self_attn: bool = False,
                 use_knowledge: bool = False,
                 use_bias: bool = True,
                 return_mean_repr: bool = False
#                 self_attention_type="dot",
#                 n_encoder_layers=3,
#                 min_std=0.01,
#                 batchnorm=False,
#                 dropout=0,
#                 attention_dropout=0,
#                 use_lvar=False,
#                 use_self_attn=False,
#                 attention_layers=2,
#                 use_lstm=False
                 ):

        super().__init__()

        self._use_self_attn = use_self_attn
        self._use_knowledge = use_knowledge
        self._return_mean_repr = return_mean_repr

        self.phi = BatchMLP(input_dim=x_dim+y_dim,
                            output_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_phi,
                            use_bias=use_bias,
                            hidden_activation=nn.ReLU(),
                            output_activation=nn.Identity())
        
        self.rho = BatchMLP(input_dim=hidden_dim,
                            output_dim=2*latent_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_rho,
                            use_bias=use_bias,
                            hidden_activation=nn.ReLU(),
                            output_activation=nn.Identity())
        if use_self_attn:
            self.self_attention_block = MultiheadAttention(input_dim=hidden_dim,
                                                           embed_dim=hidden_dim,
                                                           num_heads=8)

        if use_knowledge:
            pass
#            self.agg_mlp = BatchMLP(input_dim=hidden_dim,
#                                    ouput_dim=hidden_dim,
#                                    hidden_dim=hidden_dim,
#                                    n_h_layers=2,
#                                    use_bias=use_bias,
#                                    hidden_activation=nn.ReLU(),
#                                    output_activation=nn.Identity())

    def knowledge_aggregator(self,
                             context_repr: torch.Tensor,
                             knowledge_repr: torch.Tensor
                             ) -> torch.Tensor:

        if knowledge_repr is not None: 
            return context_repr + knowledge_repr
        else:
            return context_repr

        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                k: torch.Tensor | None = None) -> torch.distributions.normal.Normal | tuple[torch.distributions.normal.Normal, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        Returns
        -------
        q_Z : torch.distributions.normal.Normal
            Distribution q(z|x,y)
            Shape (batch_size, 1, latent_dim)
        """
    

        xy_input = torch.cat((x, y), dim=2)

        xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points, hidden_dim)
        
        if self._use_self_attn:
            xy_encoded = self.self_attention_block(xy_encoded) # Shape (batch_size, num_points, hidden_dim)
        
        mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)

        # Knowledge aggregation
        if k is not None:
            assert self._use_knowledge, "k is provided but use_knowledge is False"
        if self._use_knowledge:
            #assert k is not None, "use_knowledge is True but k is None"
            mean_repr = self.knowledge_aggregator(mean_repr, k)
        
        latent_output = self.rho(mean_repr) # Shape (batch_size, 2*latent_dim)

        mean, pre_stddev = torch.chunk(latent_output, 2, dim=-1)
        stddev = 0.1 + 0.9*F.sigmoid(pre_stddev) # Shape (batch_size, latent_dim)

        if self._return_mean_repr:
            return MultivariateNormalDiag(mean, stddev), mean_repr
        else:
            return MultivariateNormalDiag(mean, stddev) # Distribution q(z|x,y): Shape (batch_size, 1, latent_dim)



class DeterminisitcEncoder(nn.Module):
    """
    Deterministic encoder
    """
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dim: int = 128,
                 determ_dim: int = 128, # i.e. z_dim
                 n_h_layers_phi: int = 3,
                 n_h_layers_rho: int = 0,
                 use_cross_attn: bool = False,
                 use_self_attn: bool = False,
                 use_bias: bool = True
                 ):
        
        super().__init__()

        self._use_self_attn = use_self_attn
        self._use_cross_attn = use_cross_attn

        self.phi = BatchMLP(input_dim=x_dim+y_dim,
                            output_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            n_h_layers=n_h_layers_phi,
                            use_bias=use_bias,
                            hidden_activation=nn.ReLU(),
                            output_activation=nn.Identity())

        if n_h_layers_rho == 0:
            self.rho = nn.Linear(hidden_dim, determ_dim, bias=use_bias) # Equivalent to making ouput_dim=determ_dim in phi
        else:
            self.rho = BatchMLP(input_dim=hidden_dim,
                                output_dim=determ_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers_rho,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())
        
        if use_cross_attn:
            self.cross_attention_block = CrossAttention(query_dim=hidden_dim,
                                                        key_dim=hidden_dim,
                                                        value_dim=hidden_dim,
                                                        embed_dim=hidden_dim,
                                                        num_heads=8)
            self.x_to_querykey = BatchMLP(input_dim=x_dim,
                                          output_dim=hidden_dim,
                                          hidden_dim=hidden_dim,
                                          n_h_layers=2,
                                          use_bias=use_bias,
                                          hidden_activation=nn.ReLU(),
                                          output_activation=nn.Identity())
        if use_self_attn:
            pass
        
    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                x_target: torch.Tensor | None = None,
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        Returns
        -------
        r : torch.Tensor
            Shape (batch_size, determ_dim)
        """
        
        xy_input = torch.cat((x_context, y_context), dim=2)

        xy_encoded = self.phi(xy_input) # Shape (batch_size, num_points[context], hidden_dim)

        if self._use_self_attn:
            pass

        if self._use_cross_attn:
            query = self.x_to_querykey(x_target) # Shape (batch_size, num_target_points, hidden_dim)
            key = self.x_to_querykey(x_context) # Shape (batch_size, num_context_points, hidden_dim)
            value = xy_encoded # Shape (batch_size, num_context_points, hidden_dim)

            determ_output = self.cross_attention_block(query, key, value) # Shape (batch_size, num_target_points, hidden_dim)
            assert determ_output.size(1) == x_target.size(1)
            return determ_output
            
        else:
            mean_repr = torch.mean(xy_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)
            determ_output = self.rho(mean_repr)
            assert determ_output.size(1) == 1

            return determ_output # Shape (batch_size, 1, determ_dim)
            

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 x_dim: int, 
                 y_dim: int, 
                 hidden_dim: int = 128, 
                 latent_dim: int = 128,
                 determ_dim: int = 128,
                 n_h_layers: int = 4,
                 use_target_transform: bool = True,
                 use_deterministic_path: bool = False,
                 use_bias: bool = True
                 ):

        super().__init__()
        
        self._use_deterministic_path = use_deterministic_path
        self._use_target_transform = use_target_transform

        if use_target_transform:
            self.target_transform = nn.Linear(x_dim, hidden_dim)
            x_target_dim = hidden_dim
        else:
            x_target_dim = x_dim


        if use_deterministic_path:
            decoder_input_dim = x_target_dim + latent_dim + determ_dim
        else:
            decoder_input_dim = x_target_dim + latent_dim

        self.decoder = BatchMLP(input_dim=decoder_input_dim,
                                output_dim=2*y_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())

    def forward(self,
                x_target: torch.Tensor,
                z: torch.Tensor,
                r: None | torch.Tensor = None
                ) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x_target : torch.Tensor
            Shape (batch_size, num_target_points, x_dim)
        z : torch.Tensor
            Shape (batch_size, 1, latent_dim)
        r : torch.Tensor
            Shape (batch_size, 1, determ_dim) or (batch_size, num_target_points, determ_dim)
        Returns
        -------
        p_y_pred : torch.distributions.normal.Normal
            Shape (batch_size, num_target_points, y_dim)
        """
        if self._use_target_transform:
            x = self.target_transform(x_target) # Shape (batch_size, num_target_points, hidden_dim)
        else:
            x = x_target

        _, num_target_points, _ = x.size()

        assert z.size(1) == 1
        z = z.repeat(1, num_target_points, 1)

        if self._use_deterministic_path:
            assert r is not None, "use_deterministic_path is True but r is NOT PROVIDED"

            if r.size(1) == 1:
                r = r.repeat(1, num_target_points, 1)
            elif r.size(1) != num_target_points:
                raise ValueError("r must have size 1 or num_target_points in the second dimension")

            decoder_input = torch.cat((x, z, r), dim=2)

        else:
            assert r is None, "use_deterministic_path is False but r IS PROVIDED"
            decoder_input = torch.cat((x, z), dim=2)
        
        decoder_output = self.decoder(decoder_input) # Shape (batch_size, num_target_points, 2*y_dim)

        mean, pre_stddev = torch.chunk(decoder_output, 2, dim=-1)
        stddev = 0.1 + 0.9 * F.softplus(pre_stddev)

        return MultivariateNormalDiag(mean, stddev) # Shape (batch_size, num_target_points, y_dim)

class KnowledgeEncoder(nn.Module):

    def __init__(self,
                 knowledge_dim: int,
                 hidden_dim: int = 128,
                 latent_dim: int = 128,
                 n_h_layers_phi: int = 2,
                 n_h_layers_rho: int = 2,
                 only_use_linear: bool = False,
                 use_bias: bool = True
                 ) -> None:
        super().__init__()

        self._only_use_linear = only_use_linear

        if only_use_linear:
            self.linear = nn.Linear(knowledge_dim, hidden_dim, bias=use_bias)
        
        else: 
            self.phi = BatchMLP(input_dim=knowledge_dim,
                                output_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers_phi,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())
            
            self.rho = BatchMLP(input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers_rho,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())
        
    def forward(self,
                knowledge: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        knowledge : torch.Tensor
            Shape (batch_size, num_knowledge_points, knowledge_dim)
        
        Returns
        -------
        k : torch.Tensor
            Shape (batch_size, 1, hidden_dim) 
            The aggregated knowledge representation
        """
        if self._only_use_linear:
            k = self.linear(knowledge) # Shape (batch_size, 1, hidden_dim)
            assert k.size(1) == 1
        
        else:
            knowledge_encoded = self.phi(knowledge) # Shape (batch_size, num_knowledge_points, hidden_dim)

            mean_repr = torch.mean(knowledge_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)

            k = self.rho(mean_repr) # Shape (batch_size, 1, hidden_dim)

        return k # Shape (batch_size, 1, hidden_dim)



class KnowledgeDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 knowledge_dim: int, 
                 hidden_dim: int = 128, 
                 latent_dim: int = 128,
                 determ_dim: int = 128,
                 n_h_layers: int = 4,
                 use_bias: bool = True
                 ):

        super().__init__()

        self.knowledge_decoder = BatchMLP(input_dim=determ_dim,
                                output_dim=knowledge_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers,
                                use_bias=use_bias,
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Identity())

    def forward(self,
                r: torch.Tensor
                ) -> torch.Tensor:
        """
        TODO
        Parameters
        ----------
        r : torch.Tensor
            Shape (batch_size, 1, determ_dim)
        Returns
        -------
        k : torch.Tensor
            Shape (batch_size, 1, knowledge_dim)
        """
        assert r.size(1) == 1
        
        decoder_output = self.knowledge_decoder(r) # Shape (batch_size, 1, knowledge_dim)

        return decoder_output
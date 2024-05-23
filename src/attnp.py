import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .attention_modules import MultiheadAttention, CrossAttention
from .models import DeterminisitcEncoder, make_MLP
import logging
logger = logging.getLogger(__name__)

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, 
                 x_size: int = 1, 
                 y_size: int = 1, 
                 h_size: int = 128, 
                 r_size: int = 64,
                 z_size: int = 64,
                 N_h_layers: int = 3,
                 use_r: bool = False
                 ):

        super().__init__()
        
        self.use_r = use_r
        if use_r:
            xzr_to_py_layers = make_MLP(in_size=x_size+r_size+z_size,
                                       out_size=2*y_size,
                                       hidden_size=h_size,
                                       num_h_layers=N_h_layers,
                                       bias=True,
                                       hidden_acc=nn.ReLU(),
                                       output_acc=nn.Identity())
            self.xzr_to_py = nn.Sequential(*xzr_to_py_layers)

        else:
            xz_to_py_layers = make_MLP(in_size=x_size+z_size,
                                       out_size=2*y_size,
                                       hidden_size=h_size,
                                       num_h_layers=N_h_layers,
                                       bias=True,
                                       hidden_acc=nn.ReLU(),
                                       output_acc=nn.Identity())
            self.xz_to_py = nn.Sequential(*xz_to_py_layers)



    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor,
                r: None | torch.Tensor = None
                ) -> torch.distributions.normal.Normal:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_target_points, x_size)
        z : torch.Tensor
            Shape (batch_size, z_size)
        r : torch.Tensor
            Shape (batch_size, num_target_points r_size) !!!! THIS IS THE MAIN CHANGE
        Returns
        -------
        p_y_pred : torch.distributions.normal.Normal
            Shape (batch_size, num_target_points, y_size)
        """
        _, num_target_points, _ = x.size()
        z = z.unsqueeze(1).repeat(1, num_target_points, 1)

        if r is not None:
            assert self.use_r, "r provided but use_r is False"
            xzr = torch.cat((x, r, z), dim=2) # Shape (batch_size, num_target_points, x_+r_+z_size)
            py_params = self.xzr_to_py(xzr)

        else:
            xz = torch.cat((x, z), dim=2) # Shape (batch_size, num_target_points, x_+z_size)
            py_params = self.xz_to_py(xz)

        yloc, pre_yscale = torch.chunk(py_params, 2, dim=2)
        yscale = 0.1 + 0.9 * F.softplus(pre_yscale)

        return Normal(yloc, yscale)

class AttentiveNeuralProcess(nn.Module):

    def __init__(self,
                 x_dim,
                 y_dim,
                 r_dim,
                 z_dim,
                 embed_dim,
                 h_dim
                 ):
        """
        Parameters
        ----------

        """

        super().__init__()
        
        # Deterministic encoder
        deterministic_values_layers = make_MLP(in_size=x_dim+y_dim,
                                               out_size=r_dim,
                                               hidden_size=h_dim,
                                               num_h_layers=6,
                                               bias=True,
                                               hidden_acc=nn.ReLU(),
                                               output_acc=nn.Identity())
        deterministic_qks_layers = make_MLP(in_size=x_dim,
                                          out_size=embed_dim,
                                          hidden_size=h_dim,
                                          num_h_layers=2,
                                          bias=True,
                                          hidden_acc=nn.ReLU(),
                                          output_acc=nn.Identity())

        self.determinisitic_values = nn.Sequential(*deterministic_values_layers)
        self.deterministic_qks = nn.Sequential(*deterministic_qks_layers)
        self.cross_attention = CrossAttention(query_dim=embed_dim,
                                              key_dim=embed_dim,
                                              value_dim=r_dim,
                                              embed_dim=embed_dim,
                                              num_heads=8)

        # Latent encoder
        latent_context_encoder_layers = make_MLP(in_size=x_dim+y_dim,
                                                 out_size=h_dim,
                                                 hidden_size=h_dim,
                                                 num_h_layers=4,
                                                 bias=True,
                                                 hidden_acc=nn.ReLU(),
                                                 output_acc=nn.Identity())
        self.latent_context_encoder = nn.Sequential(*latent_context_encoder_layers)
        self.self_attention = MultiheadAttention(input_dim=h_dim,
                                                 embed_dim=h_dim, 
                                                 num_heads=8)
        latent_rho_encoder_layers = make_MLP(in_size=h_dim,
                                             out_size=2*z_dim,
                                             hidden_size=h_dim,
                                             num_h_layers=2,
                                             bias=True,
                                             hidden_acc=nn.ReLU(),
                                             output_acc=nn.Identity())
        self.latent_rho_encoder = nn.Sequential(*latent_rho_encoder_layers)

        self.decoder = Decoder(x_dim, y_dim, h_dim, r_dim,
                               z_dim, N_h_layers=4, use_r=False) # !!! use_r here
        
        self._init_weights()

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            #torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            torch.nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='relu')
            torch.nn.init.zeros_(m.bias)
    
    def _init_weights(self):
        self.apply(self.weights_init)
        print("Weights initialized")

    def deterministic_encoder(self, 
                              x_context: torch.Tensor,
                              y_context: torch.Tensor,
                              x_target: torch.Tensor
                              ):
        """
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context_points, x_dim)
        y_context : torch.Tensor
            Shape (batch_size, num_context_points, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target_points, x_dim)
            
        Returns
        -------


        """
        
        query = self.deterministic_qks(x_target) # Shape (batch_size, num_target_points, embed_dim)
        key = self.deterministic_qks(x_context) # Shape (batch_size, num_context_points, embed_dim)
        xy = torch.cat([x_context, y_context], dim=-1) # Shape (batch_size, num_context_points, x_dim+y_dim)
        value = self.determinisitic_values(xy) # Shape (batch_size, num_context_points, r_dim)

        r = self.cross_attention(query, key, value) # Shape (batch_size, num_target_points, embed_dim)

        return r
    
    def latent_encoder(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       knowledge: torch.Tensor | None = None,
                       ) -> torch.distributions.normal.Normal:

        xy = torch.cat([x, y], dim=-1) # Shape (batch_size, num_points, x_dim+y_dim)
        s_ = self.latent_context_encoder(xy) # Shape (batch_size, num_points, h_dim)
        #s_i = self.self_attention(s_) # Shape (batch_size, num_points, h_dim)
        s = torch.mean(s_, dim=1) # Shape (batch_size, h_dim)

        zloc, pre_zscale = torch.chunk(self.latent_rho_encoder(s), 2, dim=1)
        zscale = 0.1 + 0.9*torch.sigmoid(pre_zscale)

        q_z = torch.distributions.Normal(zloc, zscale)
        
        return q_z

    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                knowledge: torch.Tensor,
                x_target: torch.Tensor,
                y_target: None | torch.Tensor = None,
                ):

        if self.training:
            assert y_target is not None, "y_target must be provided during training"
            # assumes that x_target includes the x_context points

            q_z_context = self.latent_encoder(x_context, y_context)
            q_z_target = self.latent_encoder(x_target, y_target)
            z_target_sample = q_z_target.rsample()

            # r = self.deterministic_encoder(x_context, y_context, x_target) # Shape (batch_size, num_target_points, r_dim)

            p_y_pred = self.decoder(x=x_target, z=z_target_sample, r=None)
            return p_y_pred, q_z_target, q_z_context
        else:

            q_z_context = self.latent_encoder(x_context, y_context, knowledge)
            z_context_sample = q_z_context.rsample()

            r = self.deterministic_encoder(x_context, y_context, x_target)

            p_y_pred = self.decoder(x=x_target, z=z_context_sample, r=None) 
            return p_y_pred

    @property
    def device(self):
        return next(self.parameters()).device



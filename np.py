import torch
import torch.nn as nn
from mlp import MLP

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

class NeuralProcess(nn.Module):

    def __init__(self,
                 x_size: int,
                 y_size: int,
                 r_size: int,
                 z_size: int,
                 h_size: int,
                 num_h_layers_encoder: int,
                 num_h_layers_decoder: int
                 ):
        super().__init__() 

        self.x_size = x_size
        self.y_size = y_size
        self.r_size = r_size
        self.z_size = z_size
        self.h_size = h_size

        self.R_encoders = MLP(x_size+y_size, r_size, h_size, num_h_layers_encoder, 
                           bias=True, hidden_acc=nn.ReLU(), output_acc=nn.Identity())
        
        # representation to global latent variable 
        self.z_encoder = MLP(r_size, 2*z_size, h_size, 1, bias=True, 
                             hidden_acc=nn.ReLU(), output_acc=nn.Identity())

        self.decoder = MLP(x_size+z_size, 2*y_size, h_size, num_h_layers_decoder, 
                           bias=True, hidden_acc=nn.ReLU(), output_acc=nn.Identity())

    
    def aggregator(self, R: torch.Tensor):
        """
        Parameters
        ----------
        R : torch.Tensor
            Shape (batch_size, num_points, r_size)
        Returns
        -------
        r : torch.Tensor
            Shape (batch_size, r_size)
        """
        return torch.mean(R, dim=1)
    
    def XY_to_qZ(self, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape (batch_size, num_points, x_size)
        Y : torch.Tensor
            Shape (batch_size, num_points, y_size)

        Returns
        -------
        mu_Z, std_Z :  tuple[torch.Tensor, torch.Tensor]
            Each shape (batch_size, z_size)

        """
        XY = torch.cat((X,Y), dim=2)
        R = self.R_encoders(XY) # Shape (batch_size, num_points, r_size) TODO: check this shape
        r = self.aggregator(R) # Shape (batch_size, r_size)
        mu_Z, log_var_Z = torch.chunk(self.z_encoder(r), 2, dim=1) # Each has Shape (batch_size, z_size)

        return mu_Z, torch.exp(0.5*log_var_Z)

    def forward(self,
                X_context: torch.Tensor,
                Y_context: torch.Tensor,
                X_target: torch.Tensor,
                Y_target: None | torch.Tensor = None,
                ):
        """
        Parameters
        ----------
        X_context : torch.Tensor
            Shape (batch_size, num_context_points, x_size)

        Y_context : torch.Tensor
            Shape (batch_size, num_context_points, y_size)

        X_target : torch.Tensor
            Shape (batch_size, num_target_points, x_size)

        Y_target : torch.Tensor
            Shape (batch_size, num_target_points, y_size)

        Returns
        -------
        tuple[tuple1, tuple2, tuple3]
        y_pred_ shapes (batch_size, num_target_points+num_context_points, y_size)
        mu_Z shapes (batch_size, z_size)
        """
        batch_size, num_target_points, _ = X_target.size()
        _, num_context_points, _ = X_context.size()

        if self.training:
            X_ct = torch.cat((X_context, X_target), dim=1); Y_ct = torch.cat((Y_context, Y_target), dim=1) # full data for amortised VI

            mu_Z_context, std_Z_context = self.XY_to_qZ(X_context, Y_context)
            q_context = Normal(mu_Z_context, std_Z_context)
            mu_Z_ct, std_Z_ct = self.XY_to_qZ(X_ct, Y_ct)
            q_ct = Normal(mu_Z_ct, std_Z_ct)
            
            eps = torch.empty(mu_Z_context.size(), device=X_ct.device).normal_(0,1) # Shape (batch_size, z_size)
            Z_sample = mu_Z_ct + std_Z_ct * eps # Shape (batch_size, z_size)
            Z_sample = Z_sample.unsqueeze(1).repeat(1, num_target_points+num_context_points, 1)
            
            XZ_pred =  torch.cat((X_ct, Z_sample), dim=2) # Shape (batch_size, num_target_points+num_context_points, x_size+z_size)
            assert list(XZ_pred.shape) == [batch_size, num_target_points+num_context_points, self.x_size + self.z_size]

            mu_y_pred, log_var_y_pred = torch.chunk(self.decoder(XZ_pred), 2, dim=2) # Shape (batch_size, num_target_points+num_context_points, y_size)
            std_y_pred = torch.exp(0.5*log_var_y_pred)
            p_y_pred = Normal(mu_y_pred, std_y_pred)
            
            return p_y_pred, q_ct, q_context

        else:
            mu_Z_context, std_Z_context = self.XY_to_qZ(X_context, Y_context)
            q_context = Normal(mu_Z_context, std_Z_context)
            z_sample = q_context.rsample() # Shape (batch_size, z_size)
            z_sample = z_sample.unsqueeze(1).repeat(1, num_target_points, 1)

            XZ_pred = torch.cat((X_target, z_sample), dim=2) # Shape (batch_size, num_target_points, x_size+z_size)

            mu_y_pred, log_var_y_pred = torch.chunk(self.decoder(XZ_pred), 2, dim=2) # Shape (batch_size, num_target_points, y_size)

            std_y_pred = torch.exp(0.5*log_var_y_pred)
            p_y_pred = Normal(mu_y_pred, std_y_pred)
            
            return p_y_pred


    @staticmethod
    def KL_div_Gaussian(mu_q, sig_q, mu_p, sig_p):
        """
        KL(q || p)
        """
        kl = 0.5 * ( 2*torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2) )

        return kl.mean(dim=0).sum()

    @staticmethod
    def KL(q, p):
        return kl_divergence(q, p).mean(dim=0).sum()


if __name__ == "__main__":
    test_mlp = MLP(1, 2, 3, 4, True, nn.ReLU(), nn.ReLU())
    


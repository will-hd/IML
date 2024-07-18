import torch
import torch.nn as nn
import math
import logging
logger = logging.getLogger(__name__)

def sum_log_prob(dist, target):
    assert len(dist.batch_shape) == 3
    
    return dist.log_prob(target).sum(dim=(-1, -2)) 

class ELBOLoss(nn.Module):
    def __init__(self, reduction="mean", beta=1):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, p_yCc, q_zCc, q_zCct, y_target):
        """
        Compute the ELBO loss
        """
        if self.training:
            loss, kl, log_p = self.get_loss(p_yCc, q_zCc, q_zCct, y_target)

        else:
            raise (NotImplementedError)

        if self.reduction is None:
            return loss, kl, log_p
        elif self.reduction == "mean":
            if kl is None:
                return dict(loss=torch.mean(loss), kl=None, log_p=torch.mean(log_p), reduction="mean")
            else:
                return dict(loss=torch.mean(loss), kl=torch.mean(kl), log_p=torch.mean(log_p), reduction="mean")
        elif self.reduction == "sum":
            if kl is None:
                return dict(loss=torch.sum(loss), kl=None, log_p=torch.sum(log_p), reduction="sum")
            else:
                return dict(loss=torch.sum(loss), kl=torch.sum(kl), log_p=torch.sum(log_p), reduction="sum")
        else:
            raise (NotImplementedError)

    def get_loss(self, p_yCc, q_zCc, q_zCct, y_target):
        """
        Compute the ELBO loss during training and NLLL for validation and testing
        """
        # print(type(p_yCc))
        # print(p_yCc.batch_shape, p_yCc.event_shape) 
        # Batch shape [num_z_samples, batch_size, num_target_points]
        # Event shape [y_dim (1)]

        # print(q_zCc.batch_shape, q_zCc.event_shape) 
        
        # print(y_target.shape) # Shape [batch_size, num_target_points, y_dim]
        # print(z_samples.shape) # Shape
        if q_zCct is not None:
            # 1st term: E_{q(z | T)}[p(y_t | z)]
            sum_log_p_yCz = sum_log_prob(p_yCc, y_target)  # [num_z_samples, batch_size]
            E_z_sum_log_p_yCz = torch.mean(sum_log_p_yCz, dim=0)  # [batch_size]
            # 2nd term: KL[q(z | C, T) || q (z || C)]
            kl_z = torch.distributions.kl.kl_divergence(
                q_zCct, q_zCc
            )  # [batch_size, *n_lat]
            print(kl_z.shape)
            E_z_kl = torch.sum(kl_z, dim=1)  # [batch_size]
            loss = -(E_z_sum_log_p_yCz - self.beta * E_z_kl)
            negative_ll = -E_z_sum_log_p_yCz

        else:

            sum_log_p_yCz = sum_log_prob(p_yCc, y_target)
            sum_log_w_k = sum_log_p_yCz
            log_S_z_sum_p_y_Cz = torch.logsumexp(sum_log_w_k, 0)
            log_E_z_sum_p_yCz = log_S_z_sum_p_y_Cz - math.log(sum_log_w_k.shape[0])
            kl_z = None
            negative_ll = -log_E_z_sum_p_yCz
            loss = negative_ll
            print(loss, negative_ll)

        return loss, kl_z, negative_ll


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

    def forward(self, pred_outputs, y_target):
        """
        Compute the ELBO loss
        """
        p_yCc, q_zCc, q_zCct = pred_outputs

        if self.training:
            loss, kl, log_p = self.get_loss(p_yCc, q_zCc, q_zCct, y_target)

        else:
            raise (NotImplementedError)

        if self.reduction is None:
            return loss, kl, log_p
        elif self.reduction == "mean":
            if kl is None:
                return torch.mean(loss), None, torch.mean(log_p)
            else:
                return torch.mean(loss), torch.mean(kl), torch.mean(log_p)
        elif self.reduction == "sum":
            if kl is None:
                return torch.sum(loss), None, torch.sum(log_p)
            else:
                return torch.sum(loss), torch.sum(kl), torch.sum(log_p)
        else:
            raise (NotImplementedError)

    def get_loss(self, p_yCc, q_zCc, q_zCct, y_target):
        """
        Compute the ELBO loss during training and NLLL for validation and testing
        """
        if q_zCct is not None:
            # 1st term: E_{q(z | T)}[p(y_t | z)]
            sum_log_p_yCz = sum_log_prob(
                p_yCc, y_target
            )  # [num_z_samples, batch_size]
            E_z_sum_log_p_yCz = torch.mean(sum_log_p_yCz, dim=0)  # [batch_size]
            # 2nd term: KL[q(z | C, T) || q (z || C)]
            kl_z = torch.distributions.kl.kl_divergence(
                q_zCct, q_zCc
            )  # [batch_size, *n_lat]
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

        return loss, kl_z, negative_ll

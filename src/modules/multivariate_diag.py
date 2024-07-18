import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.independent import Independent
import logging
logger = logging.getLogger(__name__)

def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


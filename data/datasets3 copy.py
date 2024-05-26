import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from math import pi
import matplotlib.pyplot as plt

from typing import NamedTuple
from .data_generator import DataGenerator, NPRegressionDescription
import numpy as np
import torch
from torch.distributions.normal import Normal
from scipy.spatial.distance import cdist

class GPData(DataGenerator):
    '''
    Generates curves using a Gaussian Process
    '''
    def __init__(self, 
                 max_num_context: int, 
                 num_points: int = 400,
                 x_size: int = 1,
                 y_size: int = 1, 
                 l1_scale: float = 0.6,
                 sigma_scale: float = 1.0,
                 random_kernel_parameters: bool = True,
                 ):

        self.max_num_context = max_num_context
        self.num_points = num_points
        self.x_size = x_size
        self.y_size = y_size
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        self.random_kernel_parameters = random_kernel_parameters

    def _rbf_kernel(self, x, sigma_f, l1, jitter=1e-9):
        d = cdist(x, x)

        norm = (d)**2 / l1
        K = sigma_f * np.exp(-0.5 * norm)
        K += jitter * np.eye(K.shape[0])
        return K

    def _sample_GP(self, x, sigma_f, l1):
        K = self._rbf_kernel(x, sigma_f=sigma_f, l1=l1)
        mu = np.zeros(x.shape[0])
        f = np.random.multivariate_normal(mu, K)
        return f.reshape(-1, 1)

    def generate_batch(self, 
                       batch_size: int,
                       testing: bool = False,
                       override_num_context: int | None = None,
                       device: torch.device = torch.device('cpu')
                       ) -> NPRegressionDescription:
        num_context = np.random.randint(low=3, high=self.max_num_context)
        if override_num_context is not None:
            num_context = override_num_context

        if testing:
            num_target = self.num_points
            num_total_points = num_target
            x = np.linspace(-2, 2, num_target).reshape(-1, 1)
        else:
            num_target = np.random.randint(low=0, high=self.max_num_context - num_context)
            num_total_points = num_context + num_target
            x = np.random.uniform(-2, 2, num_total_points).reshape(-1, 1) 

        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            #sigma = np.random.uniform(*self.sigma_range) if self.random_kernel_parameters else self.sigma_range[1]
            sigma = np.random.rand() * (self.sigma_scale - 0.1) + 0.1
            #ls = np.random.uniform(*self.ls_range) if self.random_kernel_parameters else self.ls_range[1]
            l1 = np.random.rand() * (self.l1_scale - 0.1) + 0.1
            y = self._sample_GP(x, sigma, l1)
            
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)


        if testing:
            x_target = x_batch
            y_target = y_batch

            idx = np.random.permutation(num_target)
            x_context = x_batch[:, idx[:num_context], :]
            y_context = y_batch[:, idx[:num_context], :]

        else:
            x_target = x_batch[:, :num_target + num_context, :]
            y_target = y_batch[:, :num_target + num_context, :]

            x_context = x_batch[:, :num_context, :]
            y_context = y_batch[:, :num_context, :]


#        x_context = x_batch[:, context_idx, :]
#        y_context = y_batch[:, context_idx, :]
#
#        x_target = x_batch[:, :num_target, :]
#        y_target = y_batch[:, :num_target, :]

        return NPRegressionDescription(
            x_context=torch.from_numpy(x_context).to(torch.float32).to(device),
            y_context=torch.from_numpy(y_context).to(torch.float32).to(device),
            x_target=torch.from_numpy(x_target).to(torch.float32).to(device),
            y_target=torch.from_numpy(y_target).to(torch.float32).to(device),
            num_total_points=x_target.shape[1],
            num_context_points=num_context)

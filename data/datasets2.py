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
                 ls_range: tuple = (0.1, 0.6),
                 sigma_range: tuple = (0.1, 1.0),
                 sigma_noise: float = 2e-2,
                 random_kernel_parameters: bool = True,
                 ):

        self.max_num_context = max_num_context
        self.num_points = num_points
        self.x_size = x_size
        self.y_size = y_size
        self.ls_range = ls_range
        self.sigma_range = sigma_range
        self.sigma_noise = sigma_noise
        self.random_kernel_parameters = random_kernel_parameters

    def _rbf_kernel(self, x1, x2, sigma, ls, jitter=1e-9):
        if x2 is None:
            d = cdist(x1, x1)
        else:
            d = cdist(x1, x2)

        K = sigma * np.exp(-np.power(d, 2) / ls)
        K += jitter * np.eye(K.shape[0])
        return K

    def _sample_GP(self, x, sigma, ls):
        K = self._rbf_kernel(x, None, sigma, ls)
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
        else:
            num_target = np.random.randint(low=num_context + 1, high=100)

        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            sigma = np.random.uniform(*self.sigma_range) if self.random_kernel_parameters else self.sigma_range[1]
            ls = np.random.uniform(*self.ls_range) if self.random_kernel_parameters else self.ls_range[1]

            x = np.sort(np.random.uniform(-2, 2, num_target)).reshape(-1, 1)
            y = self._sample_GP(x, sigma, ls)
            
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        context_idx = np.random.choice(num_target, num_context, replace=False)

        x_context = x_batch[:, context_idx, :]
        y_context = y_batch[:, context_idx, :]

        x_target = x_batch[:, :num_target, :]
        y_target = y_batch[:, :num_target, :]

        return NPRegressionDescription(
            x_context=torch.from_numpy(x_context).to(torch.float32).to(device),
            y_context=torch.from_numpy(y_context).to(torch.float32).to(device),
            x_target=torch.from_numpy(x_target).to(torch.float32).to(device),
            y_target=torch.from_numpy(y_target).to(torch.float32).to(device),
            num_total_points=x_target.shape[1],
            num_context_points=num_context)

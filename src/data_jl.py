
################################################################################

import torch
import collections
import numpy as np
from torch.distributions.normal import Normal

NPRegressionDescription = collections.namedtuple(
    'NPRegressionDescription',
    ('context_x', 'context_y',
     'target_x', 'target_y',
     'num_total_points', 'num_context_points'))

class GPData():
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
                 testing: bool = False
                 ):

        self.max_num_context = max_num_context
        self.num_points = num_points
        self.x_size = x_size
        self.y_size = y_size
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        self.random_kernel_parameters = random_kernel_parameters
        self.testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        num_total_points = xdata.shape[1]

        # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2

        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2
        norm = torch.sum(norm, -1)

        kernel = sigma_f[:, :, None, None]**2 * torch.exp(-0.5 * norm)
        kernel += (sigma_noise**2) * torch.eye(
            num_total_points).expand(kernel.shape)

        return kernel

    def generate_curves(self, batch_size: int):
        num_context = torch.randint(
            low=3, high=self.max_num_context, size=(1,)).item()

        if self.testing:
            num_target = self.num_points
            num_total_points = num_target
            x_values = torch.linspace(
                -2, 2, steps=self.num_points).repeat(
                batch_size, 1).unsqueeze(-1)
        else:
            num_target = torch.randint(
                low=0, high=self.max_num_context - num_context, size=(1,)).item()
            num_total_points = num_context + num_target
            x_values = torch.rand(
                batch_size, num_total_points, self.x_size) * 4 - 2

        if self.random_kernel_parameters:
            l1 = torch.rand(
                batch_size, self.y_size, self.x_size) * (
                self.l1_scale - 0.1) + 0.1
            sigma_f = torch.rand(
                batch_size, self.y_size) * (
                self.sigma_scale - 0.1) + 0.1
        else:
            l1 = torch.ones(
                batch_size, self.y_size, self.x_size) * self.l1_scale
            sigma_f = torch.ones(
                batch_size, self.y_size) * self.sigma_scale

        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        cholesky = torch.linalg.cholesky(kernel.double()).float()

        y_values = torch.matmul(
            cholesky, 
            Normal(0, 1).sample(
                [batch_size, self.y_size, num_total_points, 1]
            )
        )
        y_values = y_values.squeeze(3).transpose(1, 2)

        if self.testing:
            target_x = x_values
            target_y = y_values

            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]
        else:
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        return NPRegressionDescription(
            context_x=context_x,
            context_y=context_y,
            target_x=target_x,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context)
        
import torch
import collections
import numpy as np
from torch.distributions.normal import Normal
from typing import NamedTuple


class NPRegressionDescription(NamedTuple):
    x_context: torch.Tensor
    y_context: torch.Tensor
    x_target: torch.Tensor
    y_target: torch.Tensor
    num_total_points: int
    num_context_points: int


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
                 sigma_noise: float = 2e-2,
                 random_kernel_parameters: bool = True,
                 ):

        self.max_num_context = max_num_context
        self.num_points = num_points
        self.x_size = x_size
        self.y_size = y_size
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        self.sigma_noise = sigma_noise
        self.random_kernel_parameters = random_kernel_parameters

    def _gaussian_kernel(self, xdata, l1, sigma_f, jitter=1e-9):
        num_total_points = xdata.shape[1]

        # make all parameters double precision
        xdata = xdata.double()
        l1 = l1.double()
        sigma_f = sigma_f.double()


        # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2

        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2
        norm = torch.sum(norm, -1)

        kernel = sigma_f[:, :, None, None]**2 * torch.exp(-0.5 * norm)
        kernel += jitter * torch.eye(num_total_points, dtype=torch.double).expand(kernel.shape) 

        #kernel += (self.sigma_noise**2) * torch.eye(
        #        num_total_points).expand(kernel.shape)

        return kernel

    def generate_batch(self, 
                       batch_size: int,
                       testing: bool = False,
                       override_num_context: int = None,
                       device: torch.device = torch.device('cpu')
                       ) -> NPRegressionDescription:
        #num_context = torch.randint(
        num_context = np.random.randint(low=3, high=self.max_num_context)
        if override_num_context is not None:
            num_context = override_num_context

        if testing:
            num_target = self.num_points
            num_total_points = num_target
            x_values = torch.linspace(
                -2, 2, steps=self.num_points).repeat(
                batch_size, 1).unsqueeze(-1)
        else:
            num_target = np.random.randint(low=0, high=self.max_num_context - num_context)
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

        jitter = 1e-9
        try: 
            kernel = self._gaussian_kernel(x_values, l1, sigma_f, jitter=jitter)

            cholesky = torch.linalg.cholesky(kernel.double()).float()
        except Exception:
            jitter = 1e-8
            while jitter < 1e-3:
                try:
                    kernel = self._gaussian_kernel(x_values, l1, sigma_f, jitter=jitter)
                    cholesky = torch.linalg.cholesky(kernel.double()).float()
                except Exception:
                    jitter *= 10

        y_values = torch.matmul(
            cholesky, 
            Normal(0, 1).sample(
                [batch_size, self.y_size, num_total_points, 1]
            )
        )
        y_values = y_values.squeeze(3).transpose(1, 2)

        if testing:
            x_target = x_values
            y_target = y_values

            idx = torch.randperm(num_target)
            x_context = x_values[:, idx[:num_context], :]
            y_context = y_values[:, idx[:num_context], :]
        else:
            x_target = x_values[:, :num_target + num_context, :]
            y_target = y_values[:, :num_target + num_context, :]

            x_context = x_values[:, :num_context, :]
            y_context = y_values[:, :num_context, :]

        return NPRegressionDescription(
            x_context=x_context.to(device),
            y_context=y_context.to(device),
            x_target=x_target.to(device),
            y_target=y_target.to(device),
            num_total_points=x_target.shape[1],
            num_context_points=num_context)
        

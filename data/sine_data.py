import torch
import numpy as np
from typing import NamedTuple


class INPRegressionDescription(NamedTuple):
    context_x: torch.Tensor
    context_y: torch.Tensor
    target_x: torch.Tensor
    target_y: torch.Tensor
    num_total_points: int
    num_context_points: int
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    
class SineData():
    '''
    Generates curves using a Gaussian Process
    '''
    def __init__(self, 
                 max_num_context: int = 10, 
                 num_target: int = 100,
                 num_test_points: int = 100,
                 x_size: int = 1,
                 y_size: int = 1, 
                 a_scale: float = 1,
                 b_scale: float = 6,
                 c_scale: float = 1,
                 random_kernel_parameters: bool = True,
                 testing: bool = False
                 ):

        self.max_num_context = max_num_context
        self.num_target = num_target
        self.num_test_points = num_test_points
        self.x_size = x_size
        self.y_size = y_size
        self.a_scale = a_scale 
        self.b_scale = b_scale
        self.c_scale = c_scale
        self.random_kernel_parameters = random_kernel_parameters
        self.testing = testing


    def generate_batch(self, 
                       batch_size: int, 
                       testing: bool = False, 
                       override_num_context: int = None,
                       device: torch.device = torch.device('cpu')
                       ) -> INPRegressionDescription:
        """
        Parameters
        ----------
        batch_size : int
            Number of curves to generate
        testing : bool
            If True, generate a fixed number of target points
        override_num_context : int
            If not None, override the number of context points
        device : torch.device
            Device to store the generated data
        """
        #num_context = torch.randint(
        num_context = np.random.randint(low=1, high=self.max_num_context)
        if override_num_context is not None:
            num_context = override_num_context

        if testing:
            num_target = self.num_test_points
            num_total_points = num_target
            x_values = torch.linspace(-2, 2, steps=self.num_test_points).repeat(batch_size, 1).unsqueeze(-1)
        else:
            num_total_points = num_context + self.num_target
            x_values = torch.rand(batch_size, num_total_points, self.x_size) * 4 - 2

        if self.random_kernel_parameters:
            a = torch.rand(batch_size, 1, self.x_size) * (2*self.a_scale) - self.a_scale
            b = torch.rand(batch_size, 1, self.x_size) * (self.b_scale - 0.1) + 0.1
            c = torch.rand(batch_size, 1, self.y_size) * (2*self.c_scale) - self.c_scale
        else:
            a = torch.ones(batch_size, 1, self.x_size) * self.a_scale
            b = torch.ones(batch_size, 1, self.x_size) * self.b_scale
            c = torch.ones(batch_size, 1, self.y_size) * self.c_scale

        y_values = a*x_values + torch.sin(b*x_values) + c

        if testing:
            target_x = x_values
            target_y = y_values

            idx = torch.randperm(self.num_test_points)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]
        else:
            target_x = x_values[:, :self.num_target + num_context, :]
            target_y = y_values[:, :self.num_target + num_context, :]

            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        return INPRegressionDescription(
            context_x=context_x.to(device),
            context_y=context_y.to(device),
            target_x=target_x.to(device),
            target_y=target_y.to(device),
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
            a=a,
            b=b,
            c=c
            )
        

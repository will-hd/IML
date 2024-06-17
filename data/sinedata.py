import torch
import numpy as np
from typing import NamedTuple

class NPRegressionDescription(NamedTuple):
    x_context: torch.Tensor
    y_context: torch.Tensor
    x_target: torch.Tensor
    y_target: torch.Tensor
    num_total_points: int
    num_context_points: int

class SineData:
    '''
    Generates data for the function f(x) = ax + bsin(x) + c
    '''
    def __init__(self, 
                 max_num_context: int, 
                 num_points: int = 400,
                 x_size: int = 1,
                 y_size: int = 1, 
                 a_range: tuple = (-1.0, 1.0),
                 b_range: tuple = (0.0, 6.0),
                 c_range: tuple = (-1.0, 1.0),
                 noise_scale: float = 0.2,
                 return_knowledge=False):
        
        self.max_num_context = max_num_context
        self.num_points = num_points
        self.x_size = x_size
        self.y_size = y_size
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.noise_scale = noise_scale
        self._return_knowledge = return_knowledge

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
            x_values = torch.linspace(-2, 2, steps=self.num_points).repeat(batch_size, 1).unsqueeze(-1)
        else:
            num_target = np.random.randint(low=0, high=self.max_num_context - num_context)
            num_total_points = num_context + num_target
            x_values = torch.rand(batch_size, num_total_points, self.x_size) * 4 - 2

        a = torch.FloatTensor(batch_size, self.y_size).uniform_(*self.a_range)
        b = torch.FloatTensor(batch_size, self.y_size).uniform_(*self.b_range)
        c = torch.FloatTensor(batch_size, self.y_size).uniform_(*self.c_range)

        y_values = a.unsqueeze(1) * x_values + torch.sin(b.unsqueeze(1)*x_values) + c.unsqueeze(1)
        y_values += torch.randn_like(y_values) * self.noise_scale

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

        if self._return_knowledge:
            return NPRegressionDescription(
                x_context=x_context.to(device),
                y_context=y_context.to(device),
                x_target=x_target.to(device),
                y_target=y_target.to(device),
                num_total_points=x_target.shape[1],
                num_context_points=num_context), (a, b, c)
        else: 
            return NPRegressionDescription(
                x_context=x_context.to(device),
                y_context=y_context.to(device),
                x_target=x_target.to(device),
                y_target=y_target.to(device),
                num_total_points=x_target.shape[1],
                num_context_points=num_context)


import torch
import collections
import numpy as np
from torch.distributions.normal import Normal
from typing import NamedTuple
import pandas as pd

from .data_generator import DataGenerator, NPRegressionDescription

# Define the CSVDataLoaderAdjusted class
class TempData:
    def __init__(self, data: pd.DataFrame, max_num_context: int):
        self.data = data
        self.max_num_context = max_num_context
        x_values = data.iloc[0][3:].values.astype('float32')
        print(x_values)
        self.x_values = torch.from_numpy(x_values).unsqueeze(0)  # Shape: [1, num_points]
        self.y_values_train = torch.tensor(data.iloc[1:508, 3:].values).float()  # Shape: [num_samples, num_points]
        self.y_values_val = torch.tensor(data.iloc[619:, 3:].values).float()  # Shape: [num_samples, num_points]
        self.y_desc = data.iloc[1:508, 2].values

    def generate_batch(self, 
                       batch_size: int, 
                       device: torch.device = torch.device('cpu'),
                       return_knowledge: bool = False
                       ) -> NPRegressionDescription:
        num_total_points = self.x_values.size(-1)
        num_context = np.random.randint(low=3, high=self.max_num_context)
        num_target = num_total_points  # Using all points as target

        # Randomly select rows excluding the zeroth row
        selected_indices = np.random.choice(self.y_values_train.size(0), batch_size, replace=False)
        selected_y_values = self.y_values_train[selected_indices]  # Shape: [batch_size, num_points]

        # Split into context and target sets
        context_indices = np.random.choice(num_total_points, num_context, replace=False)

        x_context = self.x_values[:, context_indices].repeat(batch_size, 1)  # Shape: [batch_size, num_context]
        y_context = selected_y_values[:, context_indices]  # Shape: [batch_size, num_context]

        x_target = self.x_values.repeat(batch_size, 1)  # Shape: [batch_size, num_target]
        y_target = selected_y_values  # Shape: [batch_size, num_target]

        if return_knowledge:
            knowledge = self.y_desc[selected_indices]
            return NPRegressionDescription(
                x_context=x_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, x_size]
                y_context=y_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, y_size]
                x_target=x_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, x_size]
                y_target=y_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, y_size]
                num_total_points=num_total_points,
                num_context_points=num_context
            ), knowledge

        else:
            
            return NPRegressionDescription(
                x_context=x_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, x_size]
                y_context=y_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, y_size]
                x_target=x_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, x_size]
                y_target=y_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, y_size]
                num_total_points=num_total_points,
                num_context_points=num_context
            )
 

# # Load the data
# data_path = '/content/data.csv'
# data = pd.read_csv(data_path, header=None)
# dataset = TempData(data=data, max_num_context=20)
# print(dataset.x_values)
# print(dataset.generate_batch(batch_size=16).y_context)
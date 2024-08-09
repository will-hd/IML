import torch
import collections
import numpy as np
from torch.distributions.normal import Normal
from typing import NamedTuple, Literal
from sklearn.model_selection import train_test_split
import pandas as pd

from .data_generator import DataGenerator, NPRegressionDescription

def scramble(word):
    "Shuffles the words in a sentence"
    foo = list(word)
    random.shuffle(foo)
    return ''.join(foo)

# 725 datapoints
# Define the CSVDataLoaderAdjusted class
class TempData:
    def __init__(self, data: pd.DataFrame, max_num_context: int, device):
        self.data = data
        self.max_num_context = max_num_context
        x_values = data.iloc[0][3:].values.astype('float32') # (288,)
        assert x_values.shape[0] == 288
        self.x_values = torch.linspace(-2, 2, len(x_values), device=device).unsqueeze(0)
        
        y_values = data.iloc[1:, 3:].values
        y_desc = data.iloc[1:, 2].values
        y_values_train, y_values_temp, y_desc_train, y_desc_temp = train_test_split(
            y_values, y_desc, test_size=0.3
        )
        y_values_val, y_values_test, y_desc_val, y_desc_test = train_test_split(
            y_values_temp, y_desc_temp, test_size=0.5
        )

        self.y_values_train = torch.tensor(y_values_train).float().to(device)
        self.y_values_val = torch.tensor(y_values_val).float().to(device)
        self.y_values_test = torch.tensor(y_values_test).float().to(device)
        self.y_desc_train = y_desc_train
        self.y_desc_val = y_desc_val
        self.y_desc_test = y_desc_test
        
        # # self.x_values = torch.from_numpy(x_values).unsqueeze(0).to(device)  # Shape: [1, num_points]
        
        # self.y_values_train = torch.tensor(data.iloc[1:508, 3:].values).float().to(device)  # Shape: [num_samples, num_points]
        # self.y_values_test = torch.tensor(data.iloc[509:618, 3:].values).float().to(device)  # Shape: [num_samples, num_points]
        # self.y_values_val = torch.tensor(data.iloc[619:, 3:].values).float().to(device)  # Shape: [num_samples, num_points]
        # self.y_desc_train = data.iloc[1:508, 2].values
        # self.y_desc_test = data.iloc[509:618, 2].values
        # self.y_desc_val = data.iloc[619:, 2].values

    def generate_batch(self, 
                       batch_size: int,
                       split: Literal['train', 'val', 'test'],
                       device: torch.device = torch.device('cpu'),
                       return_knowledge: bool = False,
                       num_context: None | int = None
                       ) -> NPRegressionDescription:
        num_total_points = self.x_values.size(-1)
        if num_context is None:
            num_context = np.random.randint(low=1, high=self.max_num_context)
        else:
            assert isinstance(num_context, int) 
        num_target = num_total_points  # Using all points as target

        if split == 'train':
            selected_indices = np.random.choice(self.y_values_train.size(0), batch_size, replace=False)
            selected_y_values = self.y_values_train[selected_indices]  # Shape: [batch_size, num_points]

            knowledge = self.y_desc_train[selected_indices]
        elif split == 'val':
            selected_indices = np.random.choice(self.y_values_val.size(0), batch_size, replace=False)
            selected_y_values = self.y_values_val[selected_indices]  # Shape: [batch_size, num_points]

            knowledge = self.y_desc_val[selected_indices]

        elif split == 'test':
            selected_indices = np.random.choice(self.y_values_test.size(0), batch_size, replace=False)
            selected_y_values = self.y_values_test[selected_indices]  # Shape: [batch_size, num_points]

            knowledge = self.y_desc_test[selected_indices]
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")
        # Split into context and target sets
        context_indices = np.random.choice(num_total_points // 2, num_context, replace=False)

        x_context = self.x_values[:, context_indices].repeat(batch_size, 1)  # Shape: [batch_size, num_context]
        y_context = selected_y_values[:, context_indices]  # Shape: [batch_size, num_context]

        x_target = self.x_values.repeat(batch_size, 1)  # Shape: [batch_size, num_target]
        y_target = selected_y_values  # Shape: [batch_size, num_target]
        
        if return_knowledge:
            
            return NPRegressionDescription(
                x_context=x_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, x_size]
                y_context=y_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, y_size]
                x_target=x_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, x_size]
                y_target=y_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, y_size]
                knowledge=list(knowledge), # Shape/type: TODO
                num_total_points=num_total_points,
                num_context_points=num_context
            )

        else:
            
            return NPRegressionDescription(
                x_context=x_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, x_size]
                y_context=y_context.unsqueeze(-1).to(device),  # Shape: [batch_size, num_context, y_size]
                x_target=x_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, x_size]
                y_target=y_target.unsqueeze(-1).to(device),    # Shape: [batch_size, num_target, y_size]
                knowledge=None,
                num_total_points=num_total_points,
                num_context_points=num_context
            )


# # Load the data
# data_path = '/content/data.csv'
# data = pd.read_csv(data_path, header=None)
# dataset = TempData(data=data, max_num_context=20)
# print(dataset.x_values)
# print(dataset.generate_batch(batch_size=16).y_context)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from math import pi
import matplotlib.pyplot as plt

from typing import NamedTuple


class NPRegressionDescription(NamedTuple):
    x_context: torch.Tensor | np.ndarray
    y_context: torch.Tensor | np.ndarray
    x_target: torch.Tensor | np.ndarray
    y_target: torch.Tensor | np.ndarray
    num_total_points: int
    num_context_points: int

class GPData():
    """
    self.data = [ ((x_context, y_context), (x_target, y_target)) ]
    """

    def __init__(self, sigma_range=(0.1, 1), ls_range=(0.1, 0.6), batch_size=16, max_num_context=97):

        self.sigma_range = sigma_range
        self.ls_range = ls_range
        self.batch_size = batch_size
        self.max_num_context = max_num_context
    
    def rbf_kernel(self, x1, x2, sigma, ls):
        if x2 is None:
            d = cdist(x1, x1)
        else:
            d = cdist(x1, x2)

        K = sigma*np.exp(-np.power(d, 2)/ls)
        return K

    def sample_GP(self, x, sigma, ls):
        K = self.rbf_kernel(x, None, sigma, ls)
        mu = np.zeros(x.shape)
        f = np.random.multivariate_normal(mu.flatten(), K, 1)

        return f.T
    def generate_batch(self, as_tensor: bool = False, device = None):

        num_context = np.random.randint(low=3, high=self.max_num_context)
        num_target = np.random.randint(low=num_context+1, high=100) # *includes num_context*
    
        x_batch = [] 
        y_batch = []
        for _ in range(self.batch_size):
            sigma = np.random.uniform(*self.sigma_range) 
            ls = np.random.uniform(*self.ls_range) 

            x = np.sort(np.random.uniform(-2, 2, num_target)).reshape(-1,1)
            y = self.sample_GP(x, sigma, ls)
            
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
#                context_data = (x[:num_context], y[:num_context])
#                target_data = (x[num_context:], y[num_context:])

#                batch_context.append(context_data)
#                batch_target.append(target_data)
        # make indices of length num_context
        context_idx = np.random.choice(num_target, num_context, replace=False)

        x_context = x_batch[:, context_idx, :]
        y_context = y_batch[:, context_idx, :]
        # x_context = x_batch[:, :num_context, :]
        # y_context = y_batch[:, :num_context, :]

        x_target = x_batch[:, :num_target, :]
        y_target = y_batch[:, :num_target, :]
        


        if as_tensor:
            assert device, "as_tensor = True so should specify device"
            return NPRegressionDescription(
                x_context = torch.from_numpy(x_context).to(torch.float32).to(device),
                y_context = torch.from_numpy(y_context).to(torch.float32).to(device),
                x_target = torch.from_numpy(x_target).to(torch.float32).to(device),
                y_target = torch.from_numpy(y_target).to(torch.float32).to(device),
                num_total_points=x_target.shape[1],
                num_context_points=num_context)

        else:
            return NPRegressionDescription(
                    x_context = x_context,
                    y_context=y_context,
                    x_target=x_target,
                    y_target=y_target,
                    num_total_points=x_target.shape[1],
                    num_context_points=num_context)




class GP_sine_data():
    def __init__(self, sigma_range=(0.1, 1), ls_range=(0.1, 0.6), batch_size=16, max_num_context=97):

            self.sigma_range = sigma_range
            self.ls_range = ls_range
            self.batch_size = batch_size
            self.max_num_context = max_num_context
    
    def generate_batch(self, as_tensor: bool = False, device = None):

        # num_context = np.random.randint(low=3, high=self.max_num_context)
        num_context = 4
        num_target = 4
        # num_target = np.random.randint(low=num_context+1, high=100) # *includes num_context*
    
        x_batch = [] 
        y_batch = []
        for _ in range(self.batch_size):
            sigma = np.random.uniform(*self.sigma_range) 
            ls = np.random.uniform(*self.ls_range) 

            x = np.sort(np.random.uniform(-2, 2, num_target)).reshape(-1,1)
            y = self.sample_GP(x, sigma, ls)
            
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
#                context_data = (x[:num_context], y[:num_context])
#                target_data = (x[num_context:], y[num_context:])

#                batch_context.append(context_data)
#                batch_target.append(target_data)
        # make indices of length num_context
        context_idx = np.random.choice(num_target, num_context, replace=False)

        x_context = x_batch[:, context_idx, :]
        y_context = y_batch[:, context_idx, :]
        # x_context = x_batch[:, :num_context, :]
        # y_context = y_batch[:, :num_context, :]

        x_target = x_batch[:, :num_target, :]
        y_target = y_batch[:, :num_target, :]
        

        if not as_tensor:
            return ((x_context, y_context), (x_target, y_target))

        else:
            assert device, "as_tensor = True so should specify device"
            x_context = torch.from_numpy(x_context).to(torch.float32).to(device)
            y_context = torch.from_numpy(y_context).to(torch.float32).to(device)
            x_target = torch.from_numpy(x_target).to(torch.float32).to(device)
            y_target = torch.from_numpy(y_target).to(torch.float32).to(device)

            return ((x_context, y_context), (x_target, y_target))

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":

    dataset = GPData()


    batch = dataset.generate_batch()
    
    print(batch[0][0].shape)

    for i, (x, y) in enumerate(zip(batch[0][0].squeeze(), batch[0][1].squeeze())):
        sort_idx = np.argsort(x) 
        print(sort_idx)
        plt.plot(x[sort_idx], y[sort_idx])
        plt.show()


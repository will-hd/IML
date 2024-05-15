import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from math import pi


class GPData(Dataset):
    """
    self.data = [ ((context_x, context_y), (target_x, target_y)) ]
    """

    def __init__(self, var_range=(0.1, 3), ls_range=(0.1, 2), batch_size=8,
                 num_datasets=960, max_num_context=50):

        self.num_datasets = num_datasets
        self.data = []

        var_min, var_max = var_range
        ls_min, ls_max = ls_range

        for _ in range(num_datasets // batch_size):

            num_context = np.random.randint(low=3, high=max_num_context)
            num_target = np.random.randint(low=1, high=max_num_context-num_context+1)
            print(num_context)
            print(num_target)
        
            batch_x = [] 
            batch_y = []
            for _ in range(batch_size):
                var = np.random.uniform(low=var_min, high=var_max) 
                ls = np.random.uniform(low=ls_min, high=ls_max) 

                x = np.random.uniform(-2, 2, num_context+num_target).reshape(-1,1)
                y = self.sample_GP(x, var, ls)
                
                batch_x.append(x)
                batch_y.append(y)

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            print(batch_x.shape)
#                context_data = (x[:num_context], y[:num_context])
#                target_data = (x[num_context:], y[num_context:])

#                batch_context.append(context_data)
#                batch_target.append(target_data)
            context_batch_x = batch_x[:, :num_context, :]
            context_batch_y = batch_y[:, :num_context, :]

            target_batch_x = batch_x[:, num_context:, :]
            target_batch_y = batch_y[:, num_context:, :]

            self.data.append( ((context_batch_x, context_batch_y), (target_batch_x, target_batch_y)) )
            

    def rbf_kernel(self, x1, x2, var, ls):
        if x2 is None:
            d = cdist(x1, x1)
        else:
            d = cdist(x1, x2)

        K = var*np.exp(-np.power(d, 2)/ls)
        return K

    def sample_GP(self, x, var, ls):
        K = self.rbf_kernel(x, None, var, ls)
        mu = np.zeros(x.shape)
        f = np.random.multivariate_normal(mu.flatten(), K, 1)

        return f.T

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):

        return len(self.data)

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
    print(f"Size of dataset: {len(dataset)}")


    print(dataset.data[0][0][0].shape)
    print(dataset.data[0][1][0].shape)

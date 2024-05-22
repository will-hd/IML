import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from math import pi
import matplotlib.pyplot as plt


class GPData():
    """
    self.data = [ ((context_x, context_y), (target_x, target_y)) ]
    """

    def __init__(self, sigma_range=(0.1, 1), ls_range=(0.1, 0.6), batch_size=16, max_num_context=97):

        self.sigma_range = sigma_range
        self.ls_range = ls_range
        self.batch_size = batch_size
        self.max_num_context = max_num_context
    
    def generate_batch(self, as_tensor: bool = False, device = None):

        num_context = np.random.randint(low=3, high=self.max_num_context)
        num_target = np.random.randint(low=num_context+1, high=100) # *includes num_context*
    
        batch_x = [] 
        batch_y = []
        for _ in range(self.batch_size):
            sigma = np.random.uniform(*self.sigma_range) 
            ls = np.random.uniform(*self.ls_range) 

            x = np.sort(np.random.uniform(-2, 2, num_target)).reshape(-1,1)
            y = self.sample_GP(x, sigma, ls)
            
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
#                context_data = (x[:num_context], y[:num_context])
#                target_data = (x[num_context:], y[num_context:])

#                batch_context.append(context_data)
#                batch_target.append(target_data)
        # make indices of length num_context
        context_idx = np.random.choice(num_target, num_context, replace=False)

        context_x = batch_x[:, context_idx, :]
        context_y = batch_y[:, context_idx, :]
        # context_x = batch_x[:, :num_context, :]
        # context_y = batch_y[:, :num_context, :]

        target_x = batch_x[:, :num_target, :]
        target_y = batch_y[:, :num_target, :]
        

        if not as_tensor:
            return ((context_x, context_y), (target_x, target_y))

        else:
            assert device, "as_tensor = True so should specify device"
            context_x = torch.from_numpy(context_x).to(torch.float32).to(device)
            context_y = torch.from_numpy(context_y).to(torch.float32).to(device)
            target_x = torch.from_numpy(target_x).to(torch.float32).to(device)
            target_y = torch.from_numpy(target_y).to(torch.float32).to(device)

            return ((context_x, context_y), (target_x, target_y))

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
    
        batch_x = [] 
        batch_y = []
        for _ in range(self.batch_size):
            sigma = np.random.uniform(*self.sigma_range) 
            ls = np.random.uniform(*self.ls_range) 

            x = np.sort(np.random.uniform(-2, 2, num_target)).reshape(-1,1)
            y = self.sample_GP(x, sigma, ls)
            
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
#                context_data = (x[:num_context], y[:num_context])
#                target_data = (x[num_context:], y[num_context:])

#                batch_context.append(context_data)
#                batch_target.append(target_data)
        # make indices of length num_context
        context_idx = np.random.choice(num_target, num_context, replace=False)

        context_x = batch_x[:, context_idx, :]
        context_y = batch_y[:, context_idx, :]
        # context_x = batch_x[:, :num_context, :]
        # context_y = batch_y[:, :num_context, :]

        target_x = batch_x[:, :num_target, :]
        target_y = batch_y[:, :num_target, :]
        

        if not as_tensor:
            return ((context_x, context_y), (target_x, target_y))

        else:
            assert device, "as_tensor = True so should specify device"
            context_x = torch.from_numpy(context_x).to(torch.float32).to(device)
            context_y = torch.from_numpy(context_y).to(torch.float32).to(device)
            target_x = torch.from_numpy(target_x).to(torch.float32).to(device)
            target_y = torch.from_numpy(target_y).to(torch.float32).to(device)

            return ((context_x, context_y), (target_x, target_y))

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


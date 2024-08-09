import matplotlib.pyplot as plt
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

LINE_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c',]
FILL_COLOURS = ['#A6CEE3', '#f0c39c', '#b8deb8']

def plot_predictive(model, batch, figsize=(5, 3), save=False, save_name=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''
    batch_size, _, _ = batch.x_context.shape
    assert batch_size <= 3, 'Batch size should be <= 3 for plot clarity'
    with torch.no_grad():
        # make device be the device of the model
        p_y_pred, q_z_context, q_z_target = model(batch.x_context,
                                                  batch.y_context,
                                                  batch.x_target,
                                                  batch.y_target,
                                                  batch.knowledge)
        mu = p_y_pred.mean  # Shape [num_z_samples, batch_size, num_target_points, y_dim=1]
        sigma = p_y_pred.stddev

    x_context, y_context = batch.x_context.cpu(), batch.y_context.cpu()
    x_target, y_target = batch.x_target.cpu(), batch.y_target.cpu()
    mu, sigma = mu.cpu(), sigma.cpu()
    
    plt.figure(figsize=figsize)
    for i in range(batch_size):
        plt.plot(x_target[i].flatten(), y_target[i].flatten(), 'k:')  # Plot ground truth GP
        
    
        num_z_samples = mu.shape[0]
        z_sample_idx = np.random.choice(num_z_samples, size=3)
        for j in z_sample_idx:
            plt.plot(x_target[i].flatten(), mu[j, i].flatten(), color=LINE_COLOURS[i])
            plt.fill_between(
                x_target[i].flatten(),
                mu[j, i].flatten() - sigma[j, i].flatten(),
                mu[j, i].flatten() + sigma[j, i].flatten(),
                alpha=0.3,
                facecolor=FILL_COLOURS[i],
                interpolate=True)
        plt.scatter(x_context[i].flatten(), y_context[i].flatten(), c='k')  # Plot context points
    #plt.ylim(-4, 4)
    
    plt.xlim(-2, 2)
    # print(list(x_target[0::36])+ [2.0])
    
    # Formatting the x-axis to display time in "HHMM" format
    plt.xticks(list(x_target[0].flatten()[::36])+ [2.0], labels=["0000", "0300", "0600", "0900", "1200", "1500", "1800", "2100", "2400"])
    
    # Label axes
    plt.xlabel('Time (HHMM)')
    plt.ylabel('Temperature (°C)')
    plt.ylabel('Temperature (°C)')
    if save:
        assert save_name
        plt.savefig(f'{save_name}.png', dpi=300)
    plt.show()


# def plot_sample(x_context, y_context, x_target, y_target):
    

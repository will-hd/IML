import matplotlib.pyplot as plt
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

def plot_predictive(model, batch, save=False, iter=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''

    with torch.no_grad():
        # make device be the device of the model
        p_y_pred, q_z_context, q_z_target = model(batch.x_context, batch.y_context, batch.x_target, batch.y_target, batch.knowledge)
        mu = p_y_pred.mean  # Shape [num_z_samples, batch_size, num_target_points, y_dim=1]
        sigma = p_y_pred.stddev

    x_context = batch.x_context.cpu()
    y_context = batch.y_context.cpu()
    x_target = batch.x_target.cpu()
    y_target = batch.y_target.cpu()
    mu = mu.cpu()
    sigma = sigma.cpu()
    plt.figure(figsize=(5, 3))

    plt.plot(x_target.flatten(), y_target.flatten(), 'k:')  # Plot ground truth GP
    plt.scatter(x_context.flatten(), y_context.flatten(), c='k')  # Plot context points

    num_z_samples = mu.shape[0]
    indices_to_plot = np.random.choice(num_z_samples, size=3)
    for i in indices_to_plot:
        plt.plot(x_target.flatten(), mu[i].flatten())
        plt.fill_between(
            x_target.flatten(),
            mu[i].flatten() - sigma[i].flatten(),
            mu[i].flatten() + sigma[i].flatten(),
            alpha=0.3,
            facecolor='#A6CEE3',
            interpolate=True)
    #plt.ylim(-4, 4)
    #plt.xlim(-2, 2)
    if save:
        plt.savefig(f'./results/iter_{iter}.png')
    plt.show()

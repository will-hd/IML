import matplotlib.pyplot as plt
import torch

import logging
logger = logging.getLogger(__name__)

def plot_predictive(model, batch, knowledge, save=False, iter=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''

    with torch.no_grad():
        # make device be the device of the model
        p_y_pred, q_z_context, q_z_target = model(batch.x_context, batch.y_context, batch.x_target, knowledge, batch.y_target)
        mu = p_y_pred.mean[0]
        sigma = p_y_pred.stddev[0]
        
    x_context = batch.x_context.cpu()
    y_context = batch.y_context.cpu()
    x_target = batch.x_target.cpu()
    y_target = batch.y_target.cpu()
    mu = mu.cpu()
    sigma = sigma.cpu()
    plt.figure(figsize=(5, 3))  
    # Plot ground truth GP
    plt.plot(x_target.flatten(), y_target.flatten(), 'k:')
    # Plot context points
    plt.scatter(x_context.flatten(), y_context.flatten(), c='k')
    # Plot mean of pred
    plt.plot(x_target.flatten(), mu.flatten())
    # Plot variance of pred
    plt.fill_between(
        x_target.flatten(),
        mu.flatten() - sigma.flatten(),
        mu.flatten() + sigma.flatten(),
        alpha=0.5,
        facecolor='#A6CEE3',
        interpolate=True)
    #plt.ylim(-4, 4)
    #plt.xlim(-2, 2)
    if save:
        plt.savefig(f'./results/iter_{iter}.png')
    plt.show()

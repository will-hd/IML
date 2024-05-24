import matplotlib.pyplot as plt
import torch

def plot_predictive_wassname(model, batch, save=False, iter=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''

    with torch.no_grad():
        # make device be the device of the model
        y_pred, losses, extras = model(batch.context_x, batch.context_y, batch.target_x, batch.target_y)
        mu = extras["y_dist"].mean
        sigma = extras["y_dist"].stddev
        
    context_x = batch.context_x.cpu()
    context_y = batch.context_y.cpu()
    target_x = batch.target_x.cpu()
    target_y = batch.target_y.cpu()
    mu = mu.cpu()
    sigma = sigma.cpu()
    plt.figure(figsize=(5, 3))  
    # Plot ground truth GP
    plt.plot(target_x.flatten(), target_y.flatten(), 'k:')
    # Plot context points
    plt.scatter(context_x.flatten(), context_y.flatten(), c='k')
    # Plot mean of pred
    plt.plot(target_x.flatten(), mu.flatten())
    # Plot variance of pred
    plt.fill_between(
        target_x.flatten(),
        mu.flatten() - sigma.flatten(),
        mu.flatten() + sigma.flatten(),
        alpha=0.5,
        facecolor='#A6CEE3',
        interpolate=True)
    #plt.ylim(-4, 4)
    plt.xlim(-2, 2)
    if save:
        plt.savefig(f'./results/iter_{iter}.png')
    plt.show()
def plot_predictive(model, batch, knowledge, save=False, iter=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''

    with torch.no_grad():
        # make device be the device of the model
        p_y_pred = model(batch.context_x, batch.context_y, batch.target_x, batch.target_y)
        mu = p_y_pred.mean
        sigma = p_y_pred.stddev
        
    context_x = batch.context_x.cpu()
    context_y = batch.context_y.cpu()
    target_x = batch.target_x.cpu()
    target_y = batch.target_y.cpu()
    mu = mu.cpu()
    sigma = sigma.cpu()
    plt.figure(figsize=(5, 3))  
    # Plot ground truth GP
    plt.plot(target_x.flatten(), target_y.flatten(), 'k:')
    # Plot context points
    plt.scatter(context_x.flatten(), context_y.flatten(), c='k')
    # Plot mean of pred
    plt.plot(target_x.flatten(), mu.flatten())
    # Plot variance of pred
    plt.fill_between(
        target_x.flatten(),
        mu.flatten() - sigma.flatten(),
        mu.flatten() + sigma.flatten(),
        alpha=0.5,
        facecolor='#A6CEE3',
        interpolate=True)
    #plt.ylim(-4, 4)
    plt.xlim(-2, 2)
    if save:
        plt.savefig(f'./results/iter_{iter}.png')
    plt.show()

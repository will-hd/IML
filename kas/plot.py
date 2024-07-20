import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_predictive(model, batch, save=False, iter=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''

    with torch.no_grad():
        # make device be the device of the model
        outputs = model(batch.x_context, batch.y_context, batch.x_target, batch.y_target, batch.knowledge)
        p_yCc = outputs[0]
        mu = p_yCc.mean
        sigma = p_yCc.stddev
        # mu = p_y_pred.mean  # Shape [num_z_samples, batch_size, num_target_points, y_dim=1]
        # sigma = p_y_pred.stddev

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

def kas_plot_predictive(model, x_context, y_context, x_target, y_target, knowledge, config, save=False, iter=None):
    '''
    Plot predicted mean and variance given context and targets. 
    '''

    model.training = False
    with torch.no_grad():
        # make device be the device of the model
        if config.use_knowledge:
            outputs = model(x_context, y_context, x_target, y_target, knowledge)
        else:
            outputs = model(x_context, y_context, x_target, y_target, None)
        p_yCc = outputs[0]
        mu = p_yCc.mean
        sigma = p_yCc.stddev
        # print(mu.shape)
        # print(x_context.shape, x_target.shape)
    model.training = True
    
    x_context = x_context.cpu()
    y_context = y_context.cpu()
    x_target = x_target.cpu()
    y_target = y_target.cpu()
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

# def plot_predictive(model, x_context, y_context, x_target, y_target, knowledge, config, save=False, iter=None):
#     '''
#     Plot predicted mean and variance given context and targets. 
#     '''

#     model.training = False
#     with torch.no_grad():
#         # make device be the device of the model
#         if config.use_knowledge:
#             outputs = model(x_context, y_context, x_target, y_target, knowledge)
#         else:
#             outputs = model(x_context, y_context, x_target, y_target, None)
#         p_yCc = outputs[0]
#         mu = p_yCc.mean[0]
#         sigma = p_yCc.stddev[0]
#         # print(mu.shape)
#         # print(x_context.shape, x_target.shape)
#     model.training = True
        
#     x_context = x_context.cpu()
#     y_context = y_context.cpu()
#     x_target = x_target.cpu()
#     y_target = y_target.cpu()
#     mu = mu.cpu()
#     sigma = sigma.cpu()
#     plt.figure(figsize=(5, 3))  
#     # Plot ground truth GP
#     plt.plot(x_target.flatten(), y_target.flatten(), 'k:')
#     # Plot context points
#     plt.scatter(x_context.flatten(), y_context.flatten(), c='k')
#     # Plot mean of pred
#     plt.plot(x_target.flatten(), mu.flatten())
#     # Plot variance of pred
#     plt.fill_between(
#         x_target.flatten(),
#         mu.flatten() - sigma.flatten(),
#         mu.flatten() + sigma.flatten(),
#         alpha=0.5,
#         facecolor='#A6CEE3',
#         interpolate=True)
#     #plt.ylim(-4, 4)
#     #plt.xlim(-2, 2)
#     if save:
#         plt.savefig(f'./results/iter_{iter}.png')
#     plt.show()

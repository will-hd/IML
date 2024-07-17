import matplotlib.pyplot as plt
import torch


def plot_predictive(model, x_context, y_context, x_target, y_target, knowledge, config, save=False, iter=None):
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
        mu = p_yCc.mean[0]
        sigma = p_yCc.stddev[0]
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

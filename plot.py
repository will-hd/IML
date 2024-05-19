import matplotlib.pyplot as plt

def plot_predictive(context_x, context_y, target_x, target_y, mu, sigma):
    '''
    Plot predicted mean and variance given context and targets. 
    '''
    context_x = context_x.cpu()
    context_y = context_y.cpu()
    target_x = target_x.cpu()
    target_y = target_y.cpu()
    mu = mu.cpu()
    sigma = sigma.cpu()
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
    plt.ylim(-4, 4)
    plt.show()

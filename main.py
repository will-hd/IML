import numpy as np
import torch
from torch.distributions import uniform
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from np import NeuralProcess
from datasets import GPData



def log_likelihood_Gaussian(mu_Y_pred, std_Y_pred, Y_ct):
    """

    Parameters
    ----------
    mu_Y_pred : torch.Tensor
        Shape (batch_size, num_context_points+num_target_points, y_dim)
    std_Y_pred : torch.Tensor
        Shape (batch_size, num_context_points+num_target_points, y_dim)
    Y_ct : torch.Tensor
        Shape (batch_size, num_context_points+num_target_points, y_dim)

    """
    ll = - torch.log(std_Y_pred.pow(2)) - 0.5 * ( (mu_Y_pred - Y_ct) / std_Y_pred).pow(2)

    return ll.mean(dim=0).sum()
    


if __name__ == "__main__":

    x_size = 1
    y_size = 1
    r_size = 50  # Dimension of representation of context points
    z_size = 50  # Dimension of sampled latent variable
    h_size = 50  # Dimension of hidden layers in encoder and decoder 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NeuralProcess(x_size, y_size, r_size, z_size, h_size, 2, 2).to(device)
    print(model)
    model.training = True
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)

    GP_dataset_train = GPData()

    
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0 
        for context_batch, target_batch in GP_dataset_train.data:

#            print(context_batch[0][0].shape)
#            print(target_batch[0][0].shape)
#            x = np.concatenate((context_batch[0][0], target_batch[0][0]), axis=0)
#            y = np.concatenate((context_batch[1][0], target_batch[1][0]), axis=0)
#
#            print(x.shape)
#            sort_idx = np.argsort(x.flatten())
#
#            plt.plot(x[sort_idx].flatten(), y[sort_idx].flatten())
#            plt.show()
#            
#            break
#        break
            
        
            
            X_context, Y_context = torch.Tensor(context_batch[0]).to(device), torch.Tensor(context_batch[1]).to(device)
            X_target, Y_target = torch.Tensor(target_batch[0]).to(device), torch.Tensor(target_batch[1]).to(device)

            optimiser.zero_grad()


            p_y_pred, q_ct, q_context = model(X_context, Y_context, X_target, Y_target)

            #kl = model.KL_div_Gaussian(mu_Z_ct, std_Z_ct, mu_Z_context, std_Z_context)
            kl = model.KL(q_ct, q_context)
            ll = p_y_pred.log_prob(torch.cat((Y_context, Y_target), dim=1)).mean(dim=0).sum()
            
            loss = kl - ll
            total_loss += loss.item()

            loss.backward()
            optimiser.step()
        
        print(f"Avg loss: {total_loss / len(GP_dataset_train)}")
#            print(X_context.shape)
#            print(Y_context.shape)
#
#            print(X_target.shape)
#            print(Y_target.shape)
#
        

            






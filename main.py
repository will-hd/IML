import numpy as np
import torch
from torch.distributions import uniform
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence

from np import NeuralProcess
from datasets import GPData
from plot import plot_predictive

def loss_func(p_y_pred, y_target, q_z_target, q_z_context): 
    """
    p_y_pred 
        Shape (batch_size, num_target_points, y_size)
    q_z_target
        Shape (batch_size, z_size)
    """

    log_lik = p_y_pred.log_prob(y_target).sum(-1)
    _, num_target_points = log_lik.shape
    
    kl = kl_divergence(q_z_target, q_z_context).sum(-1)
    kl = kl.unsqueeze(-1).repeat(1, num_target_points)


    return -torch.mean(log_lik - kl / num_target_points)



if __name__ == "__main__":

    x_size = 1
    y_size = 1
    h_size = 128
    z_size = 64
    r_size = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NeuralProcess(
                x_size = x_size,
                y_size = y_size,
                r_size = r_size,
                z_size = z_size,
                h_size_dec = h_size,
                h_size_enc_lat = h_size,
                h_size_enc_det = h_size,
                N_h_layers_dec = 3,
                N_xy_to_si_layers = 2,
                N_sc_to_qz_layers = 1,
                N_h_layers_enc_det = 6,
                use_r = False
                ).to(device)

    print(model)
    model.training = True
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-5)

    dataset = GPData()

    N_iterations = 100000 

    for iter in range(N_iterations):
        total_loss = 0 
        ((x_context, y_context), (x_target, y_target)) = dataset.generate_batch(batch_size=16, as_tensor=True, device=device)


        optimiser.zero_grad()

        p_y_pred, q_z_target, q_z_context = model(x_context, y_context, x_target, y_target)


        loss = loss_func(p_y_pred, y_target, q_z_target, q_z_context)

        loss.backward()
        optimiser.step()


        if iter % 5000 == 0: 
            with torch.no_grad():
                print(f"Loss {loss.item()}")

                model.training = False
                ((x_context, y_context), (x_target, y_target)) = dataset.generate_batch(batch_size=1, as_tensor=True, device=device)

                p_y_pred = model(x_context, y_context, x_target)
                plot_predictive(x_context, y_context, x_target, y_target, p_y_pred.mean, p_y_pred.stddev)
                model.training = True

        





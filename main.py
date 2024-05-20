import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

from np import NeuralProcess
from gpdata import GPData
from plot import plot_predictive
from config import Config, ConfigType
import json


def loss_function(pred_dist: Normal, 
                  target_y: torch.Tensor,
                  posterior: Normal,
                  prior: Normal):

    num_targets = target_y.size(-2)
    log_p = pred_dist.log_prob(target_y).sum(-1)
    # assert log_p.shape[-1] == 1
    # log_p = log_p.squeeze(-1)

    kl_div = torch.sum(kl_divergence(posterior, prior), dim=-1, keepdim=True)

    loss = -torch.mean(log_p - kl_div / num_targets)
    return loss



if __name__ == "__main__":
    

    with open('default_1D.json') as cf_file:
        cfg = json.load(cf_file)
        cfg = ConfigType(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NeuralProcess(
                x_size=cfg['experiment']['model']['x_size'],
                y_size=cfg['experiment']['model']['y_size'],
                r_size=cfg['experiment']['model']['r_size'],
                z_size=cfg['experiment']['model']['z_size'],
                h_size_dec=cfg['experiment']['model']['h_size_dec'],
                h_size_enc_lat=cfg['experiment']['model']['h_size_enc_lat'],
                h_size_enc_det=cfg['experiment']['model']['h_size_enc_det'],
                N_h_layers_dec=cfg['experiment']['model']['N_h_layers_dec'],
                N_xy_to_si_layers=cfg['experiment']['model']['N_xy_to_si_layers'],
                N_sc_to_qz_layers=cfg['experiment']['model']['N_sc_to_qz_layers'],
                N_h_layers_enc_det=cfg['experiment']['model']['N_h_layers_enc_det'],
                use_r=cfg['experiment']['model']['use_r'],
                ).to(device)
                

    print(model)
    model.training = True
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    LR = cfg['experiment']['training']['optimiser']['LR']
    optimiser= torch.optim.Adam(model.parameters(), lr=LR)

    N_ITERS = cfg['experiment']['training']['iterations']
    MAX_NUM_CONTEXT = cfg['experiment']['data']['max_num_context']
    dataset = GPData(MAX_NUM_CONTEXT)

    train_loss = []

    for iter in range(N_ITERS):
        total_loss = 0 
        batch = dataset.generate_curves(batch_size=16)

        context_x, context_y = batch.context_x.to(device), batch.context_y.to(device)
        target_x, target_y = batch.target_x.to(device), batch.target_y.to(device)

        optimiser.zero_grad()

        p_y_pred, q_z_target, q_z_context = model(context_x, context_y, target_x, target_y)


        loss = loss_function(p_y_pred, target_y, q_z_target, q_z_context)

        loss.backward()
        optimiser.step()
        train_loss.append(loss.item())

        if iter % 5000 == 0:
            print(f"Iter: {iter}, Loss {loss.item()}")

        if iter % 30000 == 0: 
            with torch.no_grad():
                model.training = False
                data_test = GPData(max_num_context=MAX_NUM_CONTEXT, testing=True)
                test_batch = data_test.generate_curves(batch_size=1)
                context_x, context_y = test_batch.context_x.to(device), test_batch.context_y.to(device)
                target_x, target_y = test_batch.target_x.to(device), test_batch.target_y.to(device)

                p_y_pred = model(context_x, context_y, target_x)
                plot_predictive(context_x, context_y, target_x, target_y, p_y_pred.mean, p_y_pred.stddev, save=True, iter=iter)
                model.training = True

        





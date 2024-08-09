import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import sys
sys.path.append('..')

import logging
from src.utils import setup_logging
setup_logging(console=True, file=True, debug=True, file_basename="k_sweep_BIG")
logger = logging.getLogger(__name__)

import pandas as pd
from data.tempdata import TempData
import matplotlib.pyplot as plt

from src.plot import plot_predictive
from src.informed_np import InformedNeuralProcess
from tqdm import tqdm
# from src.loss import ELBOLoss
from src.loss import ELBOLoss
from src.train import train  

from types import SimpleNamespace

def eval(model, data, use_knowledge=True, num_context=None):
    model.training = False
    with torch.no_grad():
        test_loss = 0
        n_test_batches = 128
        test_batch_size = 96
        for _ in range(n_test_batches):
            batch = data.generate_batch(batch_size=test_batch_size,
                                    device=DEVICE,
                                    return_knowledge=use_knowledge,
                                    split='test',
                                    num_context=num_context)
            
            p_y_pred, q_z_context, q_z_target = model(batch.x_context,
                                                  batch.y_context,
                                                  batch.x_target,
                                                  batch.y_target,
                                                  batch.knowledge)
            test_loss_dict = loss_function(p_y_pred, q_z_context, None, batch.y_target)
            test_loss += test_loss_dict["loss"].item() / n_test_batches
        # val_losses.append(val_loss)
        
    model.training = True
    print(test_loss)
    return test_loss

def k_and_nok_eval(model, data):
    k_test_losses = []
    nok_test_losses = []

    for num_context in range(1, 11):
        k_test_losses.append(eval(model, data, use_knowledge=True, num_context=num_context))
        nok_test_losses.append(eval(model, data, use_knowledge=False, num_context=num_context))

    return k_test_losses, nok_test_losses


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using DEVICE: {DEVICE}')
    
    USE_KNOWLEDGE = True
    logging.info(f'USE_KNOWLEDGE: {USE_KNOWLEDGE}')
    
    x_dim = 1
    y_dim = 1
    determ_dim = 128  # Dimension of representation of context points
    latent_dim = 128  # Dimension of sampled latent variable
    hidden_dim = 128  # Dimension of hidden layers in encoder and decoder

    args = dict(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                determ_dim=determ_dim,
                knowledge_dim=128,
                mlps_activation=nn.GELU(),
                x_proj_dim=1,
                n_h_layers_x_proj=0,
                n_h_layers_decoder=4,
                n_h_layers_latent_xy_encoder=3,
                n_h_layers_film_latent_encoder=3,
                path='latent',
                train_num_z_samples=4,
                test_num_z_samples=32,
                use_bias=True,
                use_context_in_target=True, # TODO investigate
                use_latent_self_attn=True,
                # use_determ_self_attn=True,
                # use_determ_cross_attn=True,
                # use_knowledge=USE_KNOWLEDGE,
                knowledge_dropout=0.3,
                roberta_return_cls=True,
                tune_llm_layer_norms=True,
                freeze_llm=True,
                knowledge_projection_n_h_layers=0,
                knowledge_aggregation_method='FiLM+MLP',
                device='cuda',
                beta=1.0
            )
    assert "use_knowledge" not in args

    
    data_path = '../data/data_with_desc.csv'
    data = pd.read_csv(data_path, header=None)
    

    AVG_LOSS_PRINT_INTERVAL = 250
    PLOT_SAMPLE_INTERVAL = 1000
    MAX_ITERS = 10

    LEARNING_RATE = 1e-3
    loss_function = ELBOLoss(beta=1, reduction='mean')
    random_states = [85, 98, 87]

    MAX_NUM_CONTEXT = 10

    #### NP ####
    np_k_test_losses = np.zeros((len(random_states), MAX_NUM_CONTEXT))
    np_nok_test_losses = np.zeros((len(random_states), MAX_NUM_CONTEXT))
    for idx, random_state in enumerate(random_states):
        data = TempData(data=data , max_num_context=MAX_NUM_CONTEXT, device=DEVICE, random_state=random_state)

        np_model = InformedNeuralProcess(
            **args,
            use_knowledge=False
        ).to(DEVICE)
        np_optimiser = torch.optim.Adam(np_model.parameters(), lr=LEARNING_RATE)
        
        np_model, best_np_model_path, optimiser, train_losses, val_losses = train(model=np_model,
                                                           data=data,
                                                           loss_function=loss_function,
                                                           optimiser=np_optimiser,
                                                           device=DEVICE,
                                                           use_knowledge=False,
                                                           max_iters=MAX_ITERS,
                                                           avg_loss_print_interval=AVG_LOSS_PRINT_INTERVAL,
                                                           plot_sample_interval=PLOT_SAMPLE_INTERVAL,
                                                           model_save_name=f"np-kdropsweep-rs-{random_state}",
                                                                                   verbose=False)
        np_model = torch.load(best_np_model_path)
        k_test_losses, nok_test_losses = k_and_nok_eval(np_model, data)
        np_k_test_losses[idx] = k_test_losses
        np_nok_test_losses[idx] = nok_test_losses

    print()
    print(np_k_text_losses.mean(axis=0), np_k_text_losses.std(axis=0))
    print(np_nok_text_losses.mean(axis=0), np_nok_text_losses.std(axis=0))

    

    #### INP ####
    inp_k_test_losses = np.zeros((5, len(random_states), MAX_NUM_CONTEXT))
    inp_nok_test_losses = np.zeros((5, len(random_states), MAX_NUM_CONTEXT))
    
    for kidx, knowledge_dropout in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        logging.info(f"Knowledge Dropout {knowledge_dropout}")
        
        for ridx, random_state in enumerate(random_states):
            data = TempData(data=data , max_num_context=MAX_NUM_CONTEXT, device=DEVICE, random_state=random_state)
            
            inp_model = InformedNeuralProcess(
                **args,
                use_knowledge=True
            ).to(DEVICE)
            inp_optimiser = torch.optim.Adam(inp_model.parameters(), lr=LEARNING_RATE)
            
            inp_model, best_inp_model_path, optimiser, train_losses, val_losses = train(model=inp_model,
                                                               data=data,
                                                               loss_function=loss_function,
                                                               optimiser=inp_optimiser,
                                                               device=DEVICE,
                                                               use_knowledge=True,
                                                               max_iters=MAX_ITERS,
                                                               avg_loss_print_interval=AVG_LOSS_PRINT_INTERVAL,
                                                               plot_sample_interval=PLOT_SAMPLE_INTERVAL,
                                                               model_save_name=f"inp-kdropsweep-{knowledge_dropout}-rs-{random_state}",
                                                                                       verbose=False)
            inp_model = torch.load(best_inp_model_path)
            k_test_losses, nok_test_losses = k_and_nok_eval(inp_model, data)
            inp_k_test_losses[kidx, ridx] = k_test_losses
            inp_nok_test_losses[kidx, ridx] = nok_test_losses
            
            print()
            print(inp_k_text_losses[kidx].mean(axis=0), inp_k_text_losses[kidx].std(axis=0))
            print(inp_nok_text_losses[kidx].mean(axis=0), inp_nok_text_losses[kidx].std(axis=0))


        
    

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import sys
sys.path.append('..')

import logging
from src.utils import setup_logging
setup_logging(console=True, file=True, debug=True, file_basename="k_sweep_1")
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
    
    data_path = '../data/data_with_desc.csv'
    data = pd.read_csv(data_path, header=None)
    data = TempData(data=data , max_num_context=10, device=DEVICE)
    
    loss_function = ELBOLoss(beta=1, reduction='mean')

    for knowledge_dropout in [0.1, 0.3, 0.5, 0.7, 0.9]:
        logging.info(f"Knowledge Dropout {knowledge_dropout}")
        model = InformedNeuralProcess(
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
                                    use_knowledge=USE_KNOWLEDGE,
                                    knowledge_dropout=knowledge_dropout,
                                    roberta_return_cls=True,
                                    tune_llm_layer_norms=True,
                                    freeze_llm=True,
                                    knowledge_projection_n_h_layers=0,
                                    knowledge_aggregation_method='FiLM+MLP',
                                    device='cuda',
                                    beta=1.0
                                    ).to(DEVICE)
            
        LEARNING_RATE = 1e-3
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        logging.info(f'Using optimiser Adam')
        
        AVG_LOSS_PRINT_INTERVAL = 250
        PLOT_SAMPLE_INTERVAL = 1000
        MAX_ITERS = 6000
        
        model, best_model_path, optimiser, train_losses, val_losses = train(model=model,
                                                           data=data,
                                                           loss_function=loss_function,
                                                           optimiser=optimiser,
                                                           device=DEVICE,
                                                           use_knowledge=USE_KNOWLEDGE,
                                                           max_iters=MAX_ITERS,
                                                           avg_loss_print_interval=AVG_LOSS_PRINT_INTERVAL,
                                                           plot_sample_interval=PLOT_SAMPLE_INTERVAL,
                                                                           model_save_name=f"inp-kdropsweep-test-{knowledge_dropout}",
                                                                               verbose=False)


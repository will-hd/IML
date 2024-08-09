import torch
import torch.nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from .plot import plot_predictive
import logging
logger = logging.getLogger(__name__)
import os


def train(model,
          data,
          loss_function,
          optimiser,
          device: torch.device | str,
          use_knowledge: bool,
          max_iters: int = 10000,
          avg_loss_print_interval: int = 1000,
          plot_sample_interval: int = 1000,
          model_save_name: None | str = None,
          verbose = True,
         ) -> tuple[torch.nn.Module, torch.optim.Optimizer, list[float], list[float]]:
    """
    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    data : TODO type
    loss_function : TODO type
    optimiser : TODO type
    device : torch.device | str
        The device to train on (and that the model and data should already be on)
    use_knowledge : bool
        Whether to use the knowledge in the model
    max_iters : int, optional
        The maximum number of iterations to train for, by default 10000
    avg_loss_print_interval : int, optional
        The interval at which to print the average loss, by default 1000
    plot_sample_interval : int, optional
        The interval at which to plot a sample, by default 1000

    Returns
    -------
    Returns the trained model, optimiser, and the training losses
    """

    train_losses = []
    val_losses = []

    for iter in range(1, max_iters+1):
        model.training = True
        optimiser.zero_grad()
    
        batch = data.generate_batch(batch_size=64,
                                    device=device,
                                    split='train',
                                    return_knowledge=use_knowledge)
        p_y_pred, q_z_context, q_z_target = model(batch.x_context,
                                                  batch.y_context,
                                                  batch.x_target,
                                                  batch.y_target,
                                                  batch.knowledge)
    
        loss_dict = loss_function(p_y_pred, q_z_context, q_z_target, batch.y_target)
    
        # old_loss, _ = calculate_loss(p_y_pred, batch.y_target, q_z_target, q_z_context)
        # old_losses.append(old_loss.item())
        
        train_loss = loss_dict["loss"]
        train_loss.backward()
        optimiser.step() 
    
        train_losses.append(train_loss.item())
        
    
        if iter % avg_loss_print_interval == 0 or iter == 1:
            if iter > 1 and verbose:
                print(f"iter {iter}: avg. Train Loss = {sum(train_losses[-avg_loss_print_interval:])/avg_loss_print_interval}")
            # print(f"iter {iter+1}: Avg. Loss = {sum(old_losses[-1000:])/1000}")
    
            with torch.no_grad():
                val_loss = 0
                n_val_batches = 128
                val_batch_size = 96
                for _ in range(n_val_batches):
                    batch = data.generate_batch(batch_size=val_batch_size,
                                                device=device,
                                                return_knowledge=use_knowledge,
                                                split='val')
                    model.training = False
                    p_y_pred, q_z_context, q_z_target = model(batch.x_context,
                                                              batch.y_context,
                                                              batch.x_target,
                                                              batch.y_target,
                                                              batch.knowledge)
                    val_loss_dict = loss_function(p_y_pred, q_z_context, None, batch.y_target)
                    val_loss += val_loss_dict["loss"].item() / n_val_batches
                val_losses.append(val_loss)
                print(f"iter {iter}: Val. Loss (NLL): {val_loss}")
                
                if iter == 1:
                    best_val_loss = val_loss
                    best_model_path = f"../exp/{str(model_save_name or '')}_iter{iter}.pt"
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"Saving new best val loss model at iter {iter}")
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if os.path.exists(best_model_path):
                            logging.info(f"Removing model at path '{best_model_path}'")
                            os.remove(best_model_path)
                            
                        best_model_path = f"../exp/{str(model_save_name or '')}_iter{iter}.pt"
                        torch.save(model.state_dict(), best_model_path)
                        logging.info(f"Saving new best val loss model at iter {iter} to path {best_model_path}")
            model.training = True
            
        if (iter % plot_sample_interval == 0 or iter == 1) and verbose:
            model.training = False
            batch = data.generate_batch(batch_size=3, device=device, return_knowledge=use_knowledge, split='train')
            plot_predictive(model, batch, save=False)
            model.training = True
    
    window = 50
    plt.plot(range(1, max_iters+1), train_losses, label='train_loss', color='#A6CEE3')
    plt.plot(range(1, len(train_losses)-window+1), [sum(train_losses[i:i+window])/window for i in range(len(train_losses)-window)],
             color='#1f77b4', label=f'train_loss_{window}_smoothed')
    # plot val at intervals of avg_loss_print_interval
    plt.plot(range(0, len(val_losses)*avg_loss_print_interval, avg_loss_print_interval),
             val_losses, label='val_loss', marker='o', markersize=5, c='#f0c39c', mec='k', mfc='#ff7f0e') # mec='#ff7f0e')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(bottom=0, top=2000)
    plt.xlim(0, len(train_losses))
    plt.show()

    return model, best_model_path, optimiser, train_losses, val_losses
    

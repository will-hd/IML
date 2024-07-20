import numpy as np
import torch
import torch.nn as nn

from typing import Callable
import time
import logging


def setup_logging(
    console: bool = True, file: bool = True, file_basename: str | None = None, debug: bool = True
) -> None:
    """Setup logging for the project.
    File logging is saved to experiments/logs folder.
    
    Parameters
    ----------
    console : bool, optional
        Whether to log to console, by default True
    file : bool, optional
        Whether to log to file, by default True
    file_basename : str, optional
        Base name for the log file to which the timestamp is joined, by default None
    debug : bool, optional
        Whether to log debug messages, by default False
    """

    timestamp = time.strftime("%d-%m-%Y-%H%M%S")
    #format_str = "[%(levelname)-8s]: %(message)s  ([%(asctime)s] %(filename)s:%(lineno)s)"
    format_str = "[%(levelname)s]: %(message)s        (%(filename)s:%(lineno)s [%(asctime)s])"
    if console:
        console_formatter = logging.Formatter(format_str, "%H:%M:%S")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG)

        logging.root.addHandler(console_handler)

    if file:
        file_formatter = logging.Formatter(format_str, "%d-%m-%Y %H:%M:%S")
        if file_basename is None:
            file_path =f"./logs/{timestamp}.log"
        else:
            file_path = f"./logs/{file_basename}_{timestamp}.log"
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        logging.root.addHandler(file_handler)

    if debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)


    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


    # Print current date and time
    logging.info(f"Logging setup completed at {timestamp}")



def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations[:num_context+num_extra_target], :]
    y_target = y[:, locations[:num_context+num_extra_target], :]
    return x_context, y_context, x_target, y_target



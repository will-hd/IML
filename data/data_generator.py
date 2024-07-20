import torch
from typing import NamedTuple
from abc import ABC, abstractmethod

class NPRegressionDescription(NamedTuple):
    x_context: torch.Tensor
    y_context: torch.Tensor
    x_target: torch.Tensor
    y_target: torch.Tensor
    knowledge: list[str] | tuple[str] | None
    num_total_points: int
    num_context_points: int


class DataGenerator(ABC):

    @abstractmethod
    def __init__(self,
                 max_num_context: int, 
                 num_test_points: int = 400,
                 x_size: int = 1,
                 y_size: int = 1
                 ):
        self.max_num_context = max_num_context
        self.num_test_points = num_test_points
        self.x_size = x_size
        self.y_size = y_size


    @abstractmethod
    def generate_batch(self,
                       batch_size: int,
                       testing: bool = False,
                       override_num_context: int | None = None,
                       device: torch.device = torch.device('cpu')
                       ) -> NPRegressionDescription:
        pass

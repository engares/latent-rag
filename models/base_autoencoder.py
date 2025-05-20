import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAutoencoder(nn.Module, ABC):
    def __init__(self, input_dim: int, latent_dim: int):
        super(BaseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

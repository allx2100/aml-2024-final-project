from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass

from abc import ABC, abstractmethod
from torch import nn

class Initializer(ABC):
    
    @abstractmethod
    def initialize(self, module: nn.Module):
        raise NotImplementedError("Must override initialize")

import numpy as np
from torch import nn

from .initializer import Initializer
from .utils.conv_delta_orthogonal_initialization import conv_delta_orthogonal_
from .utils.mean_field import MeanField


class MeanFieldInitializer(Initializer):
    def __init__(self, num_layers: int, initialization: str = "critical"):
        super().__init__()

        self.bias_mean = 0
        self.weight_mean = 0

        mean_field_calculator = MeanField(np.tanh, lambda x: 1.0 / np.cosh(x) ** 2)
        qstar = 1 / num_layers
        self.weight_var, self.bias_var = mean_field_calculator.sw_sb(qstar, 1)

        if initialization == "chaotic":
            self.weight_var *= 1.1
            self.bias_var *= 0.9
        elif initialization == "ordered":
            self.weight_var *= 0.9
            self.bias_var *= 1.1

    def initialize(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            conv_delta_orthogonal_(module.weight, np.sqrt(self.weight_var))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.bias_mean, std=np.sqrt(self.bias_var))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.weight_mean, std=np.sqrt(self.weight_var / module.in_features)
            )
            if module.bias is not None:
                module.bias.data.normal_(mean=self.bias_mean, std=np.sqrt(self.bias_var))

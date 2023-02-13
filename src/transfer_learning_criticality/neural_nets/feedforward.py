import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from ._base import _BaseNet

class FeedForwardNet(_BaseNet):

    def __init__(self, input_shape: Tuple[int, int, int], hidden_layer_width: int, num_hidden_layers: int, num_classes: int, init_weight_mean: float=0, init_weight_var: float =1, init_bias_mean: float=0, init_bias_var: float=1, non_linearity = nn.Tanh()):
        super(FeedForwardNet, self).__init__(input_shape, hidden_layer_width, num_hidden_layers, num_classes, init_weight_mean, init_weight_var, init_bias_mean, init_bias_var, non_linearity)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.init_weight_mean, std=np.sqrt(self.init_weight_var / module.in_features))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=np.sqrt(self.init_bias_var))
    
    def _create_input_layer(self):
        return nn.Linear(int(np.prod(self.input_shape)), self.hidden_layer_width)

    def _create_hidden_layers(self):
        return nn.ModuleList([nn.Linear(self.hidden_layer_width, self.hidden_layer_width) for i in range(self.num_hidden_layers - 1)])
   
    def _create_output_layer(self, num_classes):
        return nn.Linear(self.hidden_layer_width, num_classes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, int(np.prod(self.input_shape)))
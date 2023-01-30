from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple
from numpy.typing import NDArray
from ..util.conv_delta_orthogonal_initialization import conv_delta_orthogonal_


class _BaseNet(nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int], hidden_layer_width: int, num_hidden_layers: int, num_classes: int,  init_weight_mean: float=0, init_weight_var: float =1, init_bias_mean: float=0, init_bias_var: float=1, non_linearity = nn.Tanh()):
        super(_BaseNet, self).__init__()
    
        self.input_shape = input_shape
        self.hidden_layer_width = hidden_layer_width
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers > 1 else 1
        self.num_layers = self.num_hidden_layers + 2
        self.init_weight_mean = init_weight_mean
        self.init_weight_var = init_weight_var
        self.init_bias_mean = init_bias_mean
        self.init_bias_var = init_bias_var
        self.non_linearity = non_linearity

        self.input_layer: nn.Module
        self.hidden_layers: nn.ModuleList 
        self.output_layer: nn.Module 

    def forward(self, x: torch.Tensor, return_hidden_layer_activities: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, NDArray]]:
        
        if len(x.shape) != 4:
            x = x.reshape((1, *x.shape))

        out = self.input_layer(self._reshape_input(x))

        if return_hidden_layer_activities:
            hidden_layer_activities = np.zeros((self.num_hidden_layers, self.hidden_layer_width))
            hidden_layer_activities[0, :] = out.view(out.size(0), -1).cpu().numpy()

        out = self.non_linearity(out)

        for i, layer in enumerate(self.hidden_layers):
            out = layer(out)

            if return_hidden_layer_activities:
                hidden_layer_activities[i + 1, :] = out.view(out.size(0), -1).cpu().numpy()
         
            out = self.non_linearity(out)

        out = self.output_layer(out)

        if return_hidden_layer_activities:
            return out, hidden_layer_activities

        return out

    @abstractmethod
    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must override _reshape_input")

    
from abc import abstractmethod
import torch
import torch.nn as nn
from typing import Union, Tuple
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

        self.input_layer: nn.Module = self._create_input_layer()
        self.hidden_layers: nn.ModuleList = self._create_hidden_layers()
        self.output_layer: nn.Module = self._create_output_layer(num_classes)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, return_hidden_layer_activities: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if len(x.shape) != 4:
            x = x.reshape((1, *x.shape))

        out = self.input_layer(self._reshape_input(x))

        if return_hidden_layer_activities:
            hidden_layer_activities = torch.zeros((x.shape[0], self.num_hidden_layers, self.hidden_layer_width), device=x.device)
            hidden_layer_activities[:, 0, :] = out.view(out.size(0), -1)

        out = self.non_linearity(out)

        for i, layer in enumerate(self.hidden_layers):
            out = layer(out)

            if return_hidden_layer_activities:
                hidden_layer_activities[:, i + 1, :] = out.view(out.size(0), -1)
         
            out = self.non_linearity(out)

        out = self.output_layer(out)

        if return_hidden_layer_activities:
            return out, hidden_layer_activities

        return out

    def change_num_classes(self, num_classes: int):
        self.output_layer = self._create_output_layer(num_classes)

    def freeze_first_n_hidden_layers(self, n):
        self.input_layer.requires_grad_(False)
        self.hidden_layers[:n].requires_grad_(False)
    
    def reinit_last_n_hidden_layers(self, n):
        self.hidden_layers[n:].apply(self._init_weights)

    @abstractmethod
    def _init_weights(self, module: nn.Module):
        raise NotImplementedError("Must override _init_weights")

    @abstractmethod
    def _create_input_layer(self):
        raise NotImplementedError("Must override _create_input_layer")

    @abstractmethod
    def _create_hidden_layers(self):
        raise NotImplementedError("Must override _create_hidden_layers")

    @abstractmethod
    def _create_output_layer(self, num_classes: int):
        raise NotImplementedError("Must override _create_output_layer")

    @abstractmethod
    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must override _reshape_input")
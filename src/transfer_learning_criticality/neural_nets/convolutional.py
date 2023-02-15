import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
from ._base import _BaseNet
from ..util.conv_delta_orthogonal_initialization import conv_delta_orthogonal_


class ConvolutionalNet(_BaseNet):

    def __init__(self, input_shape: Tuple[int, int, int], num_conv_channels: int, num_hidden_layers: int, num_classes: int, init_weight_mean: float=0, init_weight_var: float =1, init_bias_mean: float=0, init_bias_var: float=1, non_linearity = nn.Tanh()):
        super(ConvolutionalNet, self).__init__(input_shape, num_conv_channels, num_hidden_layers, num_classes, init_weight_mean, init_weight_var, init_bias_mean, init_bias_var, non_linearity)

        self.num_conv_channels = num_conv_channels
        self.kernel_size = 3
        self.padding_size = 1

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            conv_delta_orthogonal_(module.weight, np.sqrt(self.init_weight_var))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=np.sqrt(self.init_bias_var))

    def _create_input_layer(self):
        num_in_channels = self.input_shape[0]
        input_output_size = self.input_shape[1]

        input_layers: List[nn.Module] = []
        for stride in [1, 2, 2]:
            input_layers.append(nn.Conv2d(num_in_channels, self.num_conv_channels, kernel_size=self.kernel_size, padding=self.padding_size, stride=stride))
            input_layers.append(self.non_linearity)
            num_in_channels = self.num_conv_channels
            input_output_size = int(((input_output_size - self.kernel_size + 2 * self.padding_size)/stride)+1)
        
        self.input_output_size = input_output_size
        self.hidden_layer_width = self.num_conv_channels * input_output_size * input_output_size

        return nn.Sequential(*input_layers)

    def _create_hidden_layers(self):
        return nn.ModuleList([nn.Conv2d(self.num_conv_channels, self.num_conv_channels, kernel_size=self.kernel_size, padding=self.padding_size) for i in range(self.num_hidden_layers - 1)])

    def _create_output_layer(self, num_classes):
        return nn.Sequential(nn.AvgPool2d(self.input_output_size), nn.Flatten(1), nn.Linear(self.num_conv_channels, num_classes))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x
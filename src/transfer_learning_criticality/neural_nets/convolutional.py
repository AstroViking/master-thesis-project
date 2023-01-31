import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from ._base import _BaseNet
from ..util.conv_delta_orthogonal_initialization import conv_delta_orthogonal_


class ConvolutionalNet(_BaseNet):

    def __init__(self, input_shape: Tuple[int, int, int], num_conv_channels: int, num_hidden_layers: int, num_classes: int, init_weight_mean: float=0, init_weight_var: float =1, init_bias_mean: float=0, init_bias_var: float=1, non_linearity = nn.Tanh()):
        super(ConvolutionalNet, self).__init__(input_shape, num_conv_channels, num_hidden_layers, num_classes, init_weight_mean, init_weight_var, init_bias_mean, init_bias_var, non_linearity)

        self.num_conv_channels = num_conv_channels

        num_in_channels = input_shape[0]
        input_output_size = input_shape[1]

        kernel_size=3
        padding_size=1

        input_layers: list[nn.Module] = []
        for stride in [1, 2, 2]:
            input_layers.append(nn.Conv2d(num_in_channels, num_conv_channels, kernel_size=kernel_size, padding=padding_size, stride=stride))
            input_layers.append(self.non_linearity)
            num_in_channels = num_conv_channels
            input_output_size = int(((input_output_size - kernel_size + 2 * padding_size)/stride)+1)

        self.input_layer = nn.Sequential(*input_layers)

        self.hidden_layer_width = num_conv_channels * input_output_size * input_output_size

        self.hidden_layers = nn.ModuleList([nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel_size, padding=padding_size) for i in range(self.num_hidden_layers - 1)])

        self.output_layer = nn.Sequential(nn.AvgPool2d(input_output_size), nn.Flatten(1), nn.Linear(num_conv_channels, num_classes))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            conv_delta_orthogonal_(module.weight, np.sqrt(self.init_weight_var))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=np.sqrt(self.init_bias_var))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x
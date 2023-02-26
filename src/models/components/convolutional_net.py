from typing import Tuple, List

import torch
import torch.nn as nn
from surgeon_pytorch import Inspect


class ConvolutionalNet(nn.Module):
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int], 
        num_conv_channels: int, 
        num_hidden_layers: int, 
        num_classes: int,
        non_linearity = nn.Tanh()
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_conv_channels = num_conv_channels
        self.num_hidden_layers = num_hidden_layers
        self.non_linearity = non_linearity

        num_in_channels = self.input_shape[0]
        input_output_size = self.input_shape[1]

        self.kernel_size = 3
        self.padding_size = 1

        input_layers: List[nn.Module] = []
        for stride in [1, 2, 2]:
            input_layers.append(nn.Conv2d(num_in_channels, self.num_conv_channels, kernel_size=self.kernel_size, padding=self.padding_size, stride=stride))
            input_layers.append(self.non_linearity)
            num_in_channels = self.num_conv_channels
            input_output_size = int(((input_output_size - self.kernel_size + 2 * self.padding_size)/stride)+1)
        
        self.input_output_size = input_output_size
        self.hidden_layer_width = self.num_conv_channels * input_output_size * input_output_size

        self.add_module("layer0", nn.Sequential(*input_layers))
        
        for i in range(self.num_hidden_layers - 1):
            self.add_module(f"layer{i+1}", nn.Sequential(nn.Conv2d(self.num_conv_channels, self.num_conv_channels, kernel_size=self.kernel_size, padding=self.padding_size), self.non_linearity))

        self.add_module(f"layer{self.num_hidden_layers}", nn.Linear(self.hidden_layer_width, num_classes))

    def forward(self, x: torch.Tensor):

        out = x

        for l in range(self.num_hidden_layers + 1):
            out = self.get_submodule(f"layer{l}")(out)
        
        return out
    
    def inspect(self) -> nn.Module:
        return Inspect(self, layer=[f"layer{l}" for l in range(self.num_hidden_layers)])
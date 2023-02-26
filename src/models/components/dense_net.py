from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from surgeon_pytorch import Inspect

class DenseNet(nn.Module):
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int], 
        hidden_layer_width: int, 
        num_hidden_layers: int, 
        num_classes: int,
        non_linearity = nn.Tanh()
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_layer_width = hidden_layer_width
        self.num_hidden_layers = num_hidden_layers
        self.non_linearity = non_linearity

        self.add_module("layer0", nn.Sequential(nn.Linear(int(np.prod(self.input_shape)), self.hidden_layer_width), self.non_linearity))
        
        for i in range(self.num_hidden_layers - 1):
            self.add_module(f"layer{i+1}", nn.Sequential(nn.Linear(self.hidden_layer_width, self.hidden_layer_width), self.non_linearity))

        self.add_module(f"layer{self.num_hidden_layers}", nn.Linear(self.hidden_layer_width, num_classes))

    def forward(self, x: torch.Tensor):
        batch_size, _, _, _ = x.size()
        out = x.view(batch_size, -1)

        for l in range(self.num_hidden_layers + 1):
            out = self.get_submodule(f"layer{l}")(out)

        return out
    
    def inspect(self) -> nn.Module:
        return Inspect(self, layer=[f"layer{l}" for l in range(self.num_hidden_layers)])
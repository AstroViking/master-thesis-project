import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple
from numpy.typing import NDArray

class FDNN(nn.Module):

    def __init__(self, input_size: int, hidden_layer_width: int, num_hidden_layers: int, num_classes: int, init_weight_mean: float=0, init_weight_var: float =1, init_bias_mean: float=0, init_bias_var: float=1):
        super(FDNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers > 1 else 1
        self.num_layers = self.num_hidden_layers + 2
        self.input_size = input_size
        self.input_layer = nn.Linear(input_size, hidden_layer_width) 

        self.hidden_layer_width = hidden_layer_width
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layer_width, hidden_layer_width) for i in range(self.num_hidden_layers - 1)])

        self.output_layer = nn.Linear(hidden_layer_width, num_classes)   
        self.non_linearity = nn.Tanh()

        self.init_weight_mean = init_weight_mean
        self.init_weight_var = init_weight_var
        self.init_bias_mean = init_bias_mean
        self.init_bias_var = init_bias_var

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.init_weight_mean, std=np.sqrt(self.init_weight_var / module.in_features))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=np.sqrt(self.init_bias_var))

    def forward(self, x: torch.Tensor, return_hidden_layer_activities: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, NDArray]]:

        if len(x.shape) != 4:
            x = x.reshape((1, *x.shape))

        x = x.reshape(-1, self.input_size)

        out = self.input_layer(x)

        if return_hidden_layer_activities:
            hidden_layer_activities = np.zeros((self.num_hidden_layers, self.hidden_layer_width))
            hidden_layer_activities[0, :] = out.view(out.size(1)).cpu().numpy()
        
        out = self.non_linearity(out)
        

        for i, layer in enumerate(self.hidden_layers):
            out = layer(out)

            if return_hidden_layer_activities:
                hidden_layer_activities[i + 1, :] = out.view(out.size(1)).cpu().numpy()
            
            out = self.non_linearity(out)
         
        out = out.view(out.size(0), -1)
        out = self.output_layer(out)

        if return_hidden_layer_activities:
            return out, hidden_layer_activities

        return out
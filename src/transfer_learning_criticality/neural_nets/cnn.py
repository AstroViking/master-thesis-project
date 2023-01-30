import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple
from numpy.typing import NDArray

# Taken from https://github.com/yl-1993/ConvDeltaOrthogonal-Init/blob/master/_ext/nn/init.py

# The MIT License

# Copyright (c) 2018 Lei Yang

# Permission is hereby granted, free of charge, 
# to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to 
# deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom 
# the Software is furnished to do so, 
# subject to the following conditions:

# The above copyright notice and this permission notice 
# shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
def conv_delta_orthogonal_(tensor: torch.Tensor, gain: float=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal_(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
      raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")
    
    if tensor.size(1) > tensor.size(0):
      raise ValueError("In_channels cannot be greater than out_channels.")
    
    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.linalg.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2)-1)//2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
        else:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor


class CNN(nn.Module):

    def __init__(self, input_size: int, num_in_channels: int, num_conv_channels: int, num_hidden_layers: int, num_classes: int, init_weight_mean: float=0, init_weight_var: float =1, init_bias_mean: float=0, init_bias_var: float=1, kernel_size: int=3, padding_size: int=1):
        super(CNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers > 1 else 1
        self.num_layers = self.num_hidden_layers + 2
        self.num_in_channels = num_in_channels
        self.input_size = input_size

        self.non_linearity = nn.Tanh()

        input_output_size = self.input_size
        input_layers: list[nn.Module] = []
        for stride in [1, 2, 2]:
            input_layers.append(nn.Conv2d(num_in_channels, num_conv_channels, kernel_size=kernel_size, padding=padding_size, stride=stride))
            input_layers.append(self.non_linearity)
            num_in_channels = num_conv_channels
            input_output_size = int(((input_output_size - kernel_size + 2 * padding_size)/stride)+1)

        self.input_layer = nn.Sequential(*input_layers)
        self.num_conv_channels = num_conv_channels

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel_size, padding=padding_size) for i in range(self.num_hidden_layers - 1)])

        self.hidden_layer_width = num_conv_channels * input_output_size * input_output_size

        self.pool_layer = nn.AvgPool2d(input_output_size)
        self.output_layer = nn.Linear(num_conv_channels, num_classes)

        self.init_weight_mean = init_weight_mean
        self.init_weight_var = init_weight_var
        self.init_bias_mean = init_bias_mean
        self.init_bias_var = init_bias_var

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            conv_delta_orthogonal_(module.weight, np.sqrt(self.init_weight_var))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=np.sqrt(self.init_bias_var))

    def forward(self, x: torch.Tensor, return_hidden_layer_activities: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, NDArray]]:
        
        if len(x.shape) != 4:
            x = x.reshape((1, *x.shape))

        out = self.input_layer(x)

        if return_hidden_layer_activities:
            hidden_layer_activities = np.zeros((self.num_hidden_layers, out.shape[1] * out.shape[2] * out.shape[3]))
            hidden_layer_activities[0, :] = out.view(out.size(0), -1).cpu().numpy()

        out = self.non_linearity(out)

        for i, layer in enumerate(self.hidden_layers):
            
            out = layer(out)

            if return_hidden_layer_activities:
                hidden_layer_activities[i + 1, :] = out.view(out.size(0), -1).cpu().numpy()
         
            out = self.non_linearity(out)

        out = self.pool_layer(out)
        out = out.view(out.size(0), -1)

        out = self.output_layer(out)

        if return_hidden_layer_activities:
            return out, hidden_layer_activities

        return out
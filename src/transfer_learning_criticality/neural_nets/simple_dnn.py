import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes, init_weight_mean=0, init_weight_std=1, init_bias_mean=0, init_bias_std=1):
        super(SimpleDNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers > 1 else 1
        self.input_size = input_size
        self.input_layer = nn.Linear(input_size, hidden_size) 

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for i in range(1, self.num_hidden_layers)])

        self.output_layer = nn.Linear(hidden_size, num_classes)   
        self.relu = nn.ReLU()

        self.init_weight_mean = init_weight_mean
        self.init_weight_std = init_weight_std
        self.init_bias_mean = init_bias_mean
        self.init_bias_std = init_bias_std
        
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.init_weight_mean, std=self.init_weight_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=self.init_bias_std)

    def forward(self, x):
        out = self.input_layer(x)

        for layer in self.hidden_layers:
            out = self.relu(out)
            out = layer(out)
         
        out = self.relu(out)
        out = self.output_layer(out)
        
        return out
    
    def forward_correlation(self, x1, x2):
        correlation = np.zeros(self.num_hidden_layers + 3)

        correlation[0] = torch.corrcoef(torch.stack((x1, x2)))[0,1].cpu().numpy()

        out1 = self.input_layer(x1)
        out2 = self.input_layer(x2)

        correlation[1] = torch.corrcoef(torch.stack((out1, out2)))[0,1].cpu().numpy()

        for i, layer in enumerate(self.hidden_layers):
            out1 = self.relu(out1)
            out1 = layer(out1)

            out2 = self.relu(out2)
            out2 = layer(out2)

            correlation[i+2] = torch.corrcoef(torch.stack((out1, out2)))[0,1].cpu().numpy()

         
        out1 = self.relu(out1)
        out1 = self.output_layer(out1)

        out2 = self.relu(out2)
        out2 = self.output_layer(out2)

        correlation[self.num_hidden_layers + 2] = torch.corrcoef(torch.stack((out1, out2)))[0,1].cpu().numpy()

        return correlation

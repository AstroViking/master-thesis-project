import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FDNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes, init_weight_mean=0, init_weight_std=1, init_bias_mean=0, init_bias_std=1):
        super(FDNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers > 1 else 1
        self.num_layers = self.num_hidden_layers + 2
        self.input_size = input_size
        self.input_layer = nn.Linear(input_size, hidden_size) 

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for i in range(1, self.num_hidden_layers + 1)])

        self.output_layer = nn.Linear(hidden_size, num_classes)   
        self.non_linearity = nn.Tanh()

        self.init_weight_mean = init_weight_mean
        self.init_weight_std = init_weight_std
        self.init_bias_mean = init_bias_mean
        self.init_bias_std = init_bias_std

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.init_weight_mean, std=self.init_weight_std / np.sqrt(module.in_features))
            if module.bias is not None:
                module.bias.data.normal_(mean=self.init_bias_mean, std=self.init_bias_std)

    def forward(self, x, layerwise_output=False):

        if layerwise_output:
            output_list = [x]

        out = self.input_layer(x)

        if layerwise_output:
            output_list.append(out)

        for layer in self.hidden_layers:
            out = self.non_linearity(out)
            out = layer(out)

            if layerwise_output:
                output_list.append(out)
         
        out = self.non_linearity(out)
        out = self.output_layer(out)

        if layerwise_output:
            output_list.append(out)
            return output_list

        return out

    def weight_bias_variances(self):

        weight_variances = np.zeros(self.num_layers)
        bias_variances = np.zeros(self.num_layers)
        
        i = 0
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                weight_variances[i] = np.std(layer.state_dict()['weight'].to('cpu').numpy() * np.sqrt(layer.in_features))**2
                bias_variances[i] = np.std(layer.state_dict()['bias'].to('cpu').numpy())**2
                i += 1

        return weight_variances, bias_variances

    def forward_correlations(self, x1, x2):

        activation_vectors1 = self.forward(x1, layerwise_output=True)
        activation_vectors2 = self.forward(x2, layerwise_output=True)

        n_activation_vectors = len(activation_vectors1)

        correlations = np.zeros(n_activation_vectors)
        
        for i in range(n_activation_vectors):
            correlations[i] = torch.corrcoef(torch.stack((activation_vectors1[i], activation_vectors2[i])))[0,1].to('cpu').numpy()

        return correlations
import torch


class GaussianNoiseGenerator():
   
    def __init__(self, mean, std):
        super(GaussianNoiseGenerator, self).__init__()
        self.mean = mean
        self.std = std

    def create(self, input_size):
       return torch.empty(input_size).normal_(mean=self.mean, std=self.std)


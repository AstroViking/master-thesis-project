from abc import abstractmethod
from typing import Callable, Tuple

import torch
from surgeon_pytorch import Inspect
from torch import nn


class InspectableNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], non_linearity: nn.Module = nn.Tanh()):
        super().__init__()

        self.input_layer: nn.Module = nn.Identity()
        self.hidden_layers: nn.Sequential = nn.Sequential()
        self.output_layer: nn.Module = nn.Identity()

        self.input_shape = input_shape
        self.non_linearity = non_linearity

    @abstractmethod
    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must override _reshape_input")

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_layers) + 1

    @property
    @abstractmethod
    def hidden_layer_width(self) -> int:
        raise NotImplementedError("Must override hidden_layer_width")

    @abstractmethod
    def adapt_output_layer(self, num_classes: int):
        raise NotImplementedError("Must override adapt_output_layer")

    def set_input_layer(self, layer: nn.Module):
        self.input_layer = layer

    def set_output_layer(self, layer: nn.Module):
        self.output_layer = layer

    def set_hidden_layers(self, layers: nn.Sequential):
        self.hidden_layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(self._reshape_input(x))
        out = self.hidden_layers(out)
        return self.output_layer(out)

    def inspect(self) -> Inspect:
        return Inspect(
            self,
            layer=["input_layer"]
            + [f"hidden_layers.{layer}" for layer in range(self.num_hidden_layers - 1)],
        )

    def freeze_first_n_hidden_layers(self, n):
        self.input_layer.requires_grad_(False)
        self.hidden_layers[:n].requires_grad_(False)

    def apply_last_n_hidden_layers(self, callback: Callable[[nn.Module], None], n):
        self.hidden_layers[n:].apply(callback)
        self.output_layer.apply(callback)

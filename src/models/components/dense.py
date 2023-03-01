from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from surgeon_pytorch import Inspect

from src.models.components.inspectable import InspectableNet


class DenseNet(InspectableNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        hidden_layer_width: int,
        num_hidden_layers: int,
        num_classes: int,
        non_linearity=nn.Tanh(),
    ):
        super().__init__(input_shape, num_classes, non_linearity)

        self._hidden_layer_width = hidden_layer_width

        self.set_input_layer(
            nn.Sequential(
                nn.Linear(int(np.prod(self.input_shape)), self.hidden_layer_width),
                self.non_linearity,
            )
        )
        self.set_hidden_layers(
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
                        self.non_linearity,
                    )
                    for layer in range(num_hidden_layers - 1)
                ]
            )
        )
        self.adapt_output_layer(num_classes)

    @property
    def hidden_layer_width(self) -> int:
        return self._hidden_layer_width

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

    def adapt_output_layer(self, num_classes: int):
        self.set_output_layer(torch.nn.Linear(self.hidden_layer_width, num_classes))

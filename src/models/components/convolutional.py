from typing import List, Tuple

import torch
import torch.nn as nn

from src.models.components.inspectable import InspectableNet


class ConvolutionalNet(InspectableNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_conv_channels: int,
        num_hidden_layers: int,
        num_classes: int,
        non_linearity=nn.Tanh(),
    ):
        super().__init__(input_shape, non_linearity)

        num_in_channels = self.input_shape[0]
        input_output_size = self.input_shape[1]
        kernel_size = 3
        padding_size = 1
        input_layers: List[nn.Module] = []

        for stride in [1, 2, 2]:
            input_layers.append(
                nn.Conv2d(
                    num_in_channels,
                    num_conv_channels,
                    kernel_size=kernel_size,
                    padding=padding_size,
                    stride=stride,
                )
            )
            input_layers.append(self.non_linearity)
            num_in_channels = num_conv_channels
            input_output_size = int(
                ((input_output_size - kernel_size + 2 * padding_size) / stride) + 1
            )

        self._input_output_size = input_output_size
        self._num_conv_channels = num_conv_channels
        self._hidden_layer_width = num_conv_channels * input_output_size * input_output_size

        self.set_input_layer(nn.Sequential(*input_layers))
        self.set_hidden_layers(
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            num_conv_channels,
                            num_conv_channels,
                            kernel_size=kernel_size,
                            padding=padding_size,
                        ),
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
        return x

    def adapt_output_layer(self, num_classes: int):
        self.set_output_layer(
            nn.Sequential(
                nn.AvgPool2d(self._input_output_size),
                nn.Flatten(1),
                nn.Linear(self._num_conv_channels, num_classes),
            )
        )

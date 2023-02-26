from typing import Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .figure import Figure


class WeightBiasVariance(Figure):
    def __init__(self, title: str, weight_variances: np.ndarray, bias_variances: np.ndarray):
        super().__init__()

        self.set_subplots(rows=2, cols=2, vertical_spacing=0.2, horizontal_spacing=0.2)

        self.add_trace(
            go.Scatter(
                x=[x for x in range(len(weight_variances))],
                y=weight_variances,
                name="Weight",
            ),
            row=1,
            col=1,
        )
        self.update_xaxes(title_text="$l$", row=1, col=1)
        self.update_yaxes(
            # range=[0, 2],
            title_text=r"$\sigma_w^2$",
            row=1,
            col=1,
        )

        self.add_trace(
            go.Scatter(x=[x for x in range(len(bias_variances))], y=bias_variances, name="Bias"),
            row=1,
            col=2,
        )
        self.update_xaxes(title_text="$l$", row=1, col=2)
        self.update_yaxes(
            # range=[0, 2],
            title_text=r"$\sigma_b^2$",
            row=1,
            col=2,
        )

        self.update_layout(autosize=True, height=900, width=1600, title_text=title)

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .figure import Figure


class Accuracy(Figure):
    def __init__(self, title: str, metrics_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        super().__init__()

        self.set_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=["Train accuracy", "Test accuracy"],
        )

        for label, metrics in metrics_dict.items():

            epoch_indices = metrics.index.unique()

            self.add_trace(
                go.Scatter(
                    x=epoch_indices,
                    y=metrics.loc[:, ("Train", "Accuracy")],
                    name=label,
                    mode="lines+markers",
                    legendgroup="train",
                ),
                row=1,
                col=1,
            )

            self.add_trace(
                go.Scatter(
                    x=epoch_indices,
                    y=metrics.loc[:, ("Test", "Accuracy")],
                    name=label,
                    mode="lines+markers",
                    legendgroup="test",
                ),
                row=1,
                col=2,
            )

        self.update_xaxes(title_text="Epoch", row=1, col=1)
        self.update_yaxes(title_text="$Accuracy$", row=1, col=1)

        self.update_xaxes(title_text="Epoch", row=1, col=2)
        self.update_yaxes(title_text="$Accuracy$", row=1, col=2)

        self.update_layout(height=900, width=1600, title_text=title, legend_tracegroupgap=20)

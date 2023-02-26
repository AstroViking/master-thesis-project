from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .figure import Figure


class Correlation(Figure):
    def __init__(
        self, title: str, correlations_dict: Dict[str, pd.DataFrame], show_error_bars: bool = True
    ):
        super().__init__()

        self.set_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=["Same class", "Different class"],
        )

        for label, correlations in correlations_dict.items():

            layer_indices = correlations.columns.unique(level="Layer")

            self.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=correlations.loc["Same class", (slice(None), "Correlation")],
                    name=label,
                    error_y=dict(
                        type="data",
                        array=correlations.loc["Same class", (slice(None), "Error")],
                        visible=True,
                    )
                    if show_error_bars
                    else None,
                    mode="lines+markers",
                    legendgroup="same",
                ),
                row=1,
                col=1,
            )

            self.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=correlations.loc["Different class", (slice(None), "Correlation")],
                    name=label,
                    error_y=dict(
                        type="data",
                        array=correlations.loc["Different class", (slice(None), "Error")],
                        visible=True,
                    )
                    if show_error_bars
                    else None,
                    mode="lines+markers",
                    legendgroup="different",
                ),
                row=1,
                col=2,
            )

        self.update_xaxes(title_text="$l$", row=1, col=1, automargin=True)

        self.update_yaxes(
            range=[-0.2, 1.2], title_text=r"$\hat{c}(x1, x1)$", row=1, col=1, automargin=True
        )

        self.update_xaxes(title_text="$l$", row=1, col=2, automargin=True)

        self.update_yaxes(
            range=[-0.2, 1.2], title_text=r"$\hat{c}(x1, x2)$", row=1, col=2, automargin=True
        )

        self.update_layout(
            autosize=True, title_text=title, width=1600, height=900, legend_tracegroupgap=20
        )

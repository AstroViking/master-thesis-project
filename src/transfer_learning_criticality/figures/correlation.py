from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ._default import set_default_layout


def average_correlation_same_vs_different_class(title: str, correlations_dict: Dict[str, pd.DataFrame], show_error_bars: bool=True) -> go.Figure:

    figure = make_subplots(
        rows=1, 
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=[
            "Same class",
            "Different class"
        ]
    )

    for label, correlations in correlations_dict.items():

        layer_indices = correlations.columns.unique(level="Layer")

        figure.add_trace(
            go.Scatter(
                x=layer_indices, 
                y=correlations.loc["Same class", (slice(None), "Correlation")], 
                name=label,
                error_y=dict(
                    type='data',
                    array=correlations.loc["Same class", (slice(None), "Error")],
                    visible=True
                ) if show_error_bars else None,
                mode="lines+markers",
                legendgroup="same"
            ),
            row=1, 
            col=1
        )

        figure.add_trace(
            go.Scatter(
                x=layer_indices, 
                y=correlations.loc["Different class", (slice(None), "Correlation")], 
                name=label,
                error_y=dict(
                    type='data',
                    array=correlations.loc["Different class", (slice(None), "Error")],
                    visible=True
                ) if show_error_bars else None,
                mode="lines+markers",
                legendgroup="different"
            ),
            row=1, 
            col=2
        )

    figure.update_xaxes(
        title_text="$l$", 
        row=1, 
        col=1,
        automargin=True
    )

    figure.update_yaxes(
        range=[-0.2, 1.2],
        title_text="$\hat{c}(x1, x1)$", 
        row=1, 
        col=1,
        automargin=True
    )

    figure.update_xaxes(
        title_text="$l$", 
        row=1, 
        col=2,
        automargin=True
    )

    figure.update_yaxes(
        range=[-0.2, 1.2],
        title_text="$\hat{c}(x1, x2)$", 
        row=1, 
        col=2,
        automargin=True
    )

    figure.update_layout(
        autosize=True,
        title_text=title,
        width=1600,
        height=900,
        legend_tracegroupgap = 20
    )

    return set_default_layout(figure)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ._default import set_default_layout


def average_correlation_same_vs_different_class(title: str, correlations_dict: dict[str, pd.DataFrame]) -> go.Figure:

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
                ),
                mode="markers"
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
                ),
                mode="markers"
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
        height=900
    )

    return set_default_layout(figure)


def average_correlation_between_classes(title: str, correlations: pd.DataFrame, n_cols: int = 2) -> go.Figure:

    n_rows = int(len(correlations.index)/n_cols) + 1

    figure = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        horizontal_spacing=0.5/n_cols,
        vertical_spacing=0.5/n_rows
    )

    current_row = 1
    current_col = 1

    for idx in correlations.index:

        figure.add_trace(
            go.Scatter(
                x=correlations.columns.unique(level="Layer"), 
                y=correlations.loc[idx, (slice(None), "Correlation")].to_numpy(), 
                name=f"Class {idx[0]} vs Class {idx[1]}",
                error_y=dict(
                    type='data',
                    array=correlations.loc[idx, (slice(None), "Error")].to_numpy(),
                    visible=True
                ),
                mode="markers"
            ),
            row=current_row, 
            col=current_col
        )
        figure.update_xaxes(
            title_text="$l$", 
            row=current_row, 
            col=current_col,
            automargin=True
        )
        figure.update_yaxes(
            range=[-1.2, 1.2],
            title_text="$\hat{c}(x1, x2)$", 
            row=current_row, 
            col=current_col,
            automargin=True
        )

        if current_col + 1 <= n_cols:
            current_col += 1
        else:
            current_col = 1
            current_row += 1

    figure.update_layout(
        autosize=True,
        title_text=title,
        width=1500 * n_cols/n_rows + 600,
        height=1500
    )

    return set_default_layout(figure)
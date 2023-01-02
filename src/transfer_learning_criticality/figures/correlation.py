import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def average_correlation_figure(title, correlations: pd.DataFrame, n_cols = 2):

    n_categories = len(correlations.index.unique(level='Category'))
    n_rows = int(n_categories/n_cols) + 1

    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        vertical_spacing=0.2,
        horizontal_spacing=0.2
    )

    current_row = 1
    current_col = 1

    for category in correlations.index.unique(level='Category'):

        fig.add_trace(
            go.Scatter(
                x=correlations.columns, 
                y=correlations.loc[category, :].mean(axis=0), 
                name=category,
            ),
            row=current_row, 
            col=current_col
        )
        fig.update_xaxes(
            title_text="$l$", 
            row=current_row, 
            col=current_col
        )
        fig.update_yaxes(
            range=[-1.2, 1.2],
            title_text="$\hat{c}(x_1, x_2)$", 
            row=current_row, 
            col=current_col
        )

        if current_col + 1 <= n_cols:
            current_col += 1
        else:
            current_col = 1
            current_row += 1

    fig.update_layout(
        height=1000, 
        width=800, 
        title_text=title
    )

    return fig
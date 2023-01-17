import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement


def average_correlation(title: str, activities: pd.DataFrame, n_cols: int = 2) -> go.Figure:

    class_indices = activities.index.unique(level="Class")
    layer_indices = activities.columns.unique(level="Layer")
    sample_indices = activities.index.unique(level="Sample")

    n_samples = len(sample_indices)
    n_layers = len(layer_indices)

    class_combinations = list(combinations_with_replacement(class_indices, 2))
    n_rows = int(len(class_combinations)/n_cols) + 1

    figure = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        horizontal_spacing=0.5/n_cols,
        vertical_spacing=0.5/n_rows
    )

    current_row = 1
    current_col = 1

    for c1, c2 in class_combinations:

        correlations = np.zeros(n_layers)

        for l in layer_indices:
            for s in sample_indices:
                random_sample_index = np.random.choice(np.delete(sample_indices, s), size=1)[0]
                correlations[l] += np.corrcoef(activities.loc[(c1, s), l].to_numpy().astype(np.float64), activities.loc[(c2, random_sample_index), l].to_numpy().astype(np.float64))[0][1]
            
            correlations[l] /= n_samples

        figure.add_trace(
            go.Scatter(
                x=layer_indices, 
                y=correlations, 
                name=f"Class {c1} vs Class {c2}",
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
        width=1500,
        height=1500
    )

    return figure
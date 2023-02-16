from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from numpy.typing import NDArray

from ._default import set_default_layout


def model_accuracy_vs_epoch(title: str, metrics_dict: Dict[str, pd.DataFrame]) -> go.Figure:

    figure = make_subplots(
        rows=2, 
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=[
            "Train accuracy",
            "Test accuracy"
        ]
    )

    for label, metrics in  metrics_dict.items():

        epoch_indices = metrics.index.unique()

        figure.add_trace(
            go.Scatter(
                x=epoch_indices, 
                y=metrics.loc[:, ("Train", "Accuracy")],
                name=label,
                mode="lines+markers",
                legendgroup="train"
            ),
            row=1, 
            col=1
        )

        figure.add_trace(
            go.Scatter(
                x=epoch_indices, 
                y=metrics.loc[:, ("Test", "Accuracy")],
                name=label,
                mode="lines+markers",
                legendgroup="test"
            ),
            row=1, 
            col=2
        )


    figure.update_xaxes(
        title_text="Epoch", 
        row=1, 
        col=1
    )
    figure.update_yaxes(
        title_text="$Accuracy$", 
        row=1, 
        col=1
    )
    
    figure.update_xaxes(
        title_text="Epoch", 
        row=1, 
        col=2
    )
    figure.update_yaxes(
        title_text="$Accuracy$", 
        row=1, 
        col=2
    )

    figure.update_layout(
        height=900, 
        width=1600, 
        title_text=title,
        legend_tracegroupgap = 20
    )

    return set_default_layout(figure)


def model_weight_bias_variance(title: str, weight_variances: NDArray, bias_variances: NDArray) -> go.Figure:

    figure = make_subplots(
        rows=2, 
        cols=2,
        vertical_spacing=0.2,
        horizontal_spacing=0.2
    )

    figure.add_trace(
        go.Scatter(
            x=[x for x in range(len(weight_variances))], 
            y=weight_variances,
            name="Weight",
        ),
        row=1, 
        col=1
    )
    figure.update_xaxes(
        title_text="$l$", 
        row=1, 
        col=1
    )
    figure.update_yaxes(
        #range=[0, 2],
        title_text="$\sigma_w^2$", 
        row=1, 
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=[x for x in range(len(bias_variances))], 
            y=bias_variances,
            name="Bias"
        ),
        row=1, 
        col=2
    )
    figure.update_xaxes(
        title_text="$l$", 
        row=1, 
        col=2
    )
    figure.update_yaxes(
        #range=[0, 2],
        title_text="$\sigma_b^2$",  
        row=1, 
        col=2
    )

    figure.update_layout(
        autosize=True,
        height=900, 
        width=1600, 
        title_text=title
    )

    return set_default_layout(figure)
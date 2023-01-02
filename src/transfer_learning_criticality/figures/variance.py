import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def layer_variance_figure(title, weight_variances, bias_variances):

    fig = make_subplots(
        rows=2, 
        cols=2,
        vertical_spacing=0.2,
        horizontal_spacing=0.2
    )

    fig.add_trace(
        go.Scatter(
            x=[x for x in range(len(weight_variances))], 
            y=weight_variances,
            name='Weight',
        ),
        row=1, 
        col=1
    )
    fig.update_xaxes(
        title_text="$l$", 
        row=1, 
        col=1
    )
    fig.update_yaxes(
        #range=[0, 2],
        title_text="$\sigma_w^2$", 
        row=1, 
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[x for x in range(len(bias_variances))], 
            y=bias_variances,
            name='Bias'
        ),
        row=1, 
        col=2
    )
    fig.update_xaxes(
        title_text="$l$", 
        row=1, 
        col=2
    )
    fig.update_yaxes(
        #range=[0, 2],
        title_text="$\sigma_b^2$",  
        row=1, 
        col=2
    )

    fig.update_layout(
        height=1000, 
        width=800, 
        title_text=title
    )

    return fig
from typing import Dict
import plotly.graph_objects as go
from sklearn.metrics import davies_bouldin_score
import numpy as np
import pandas as pd

from ._default import set_default_layout


def davies_bouldin_index(title: str, activities_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    
    figure = go.Figure()

    for label, activities in activities_dict.items():

        n_layers = len(activities.columns.unique(level="Layer"))
        db_index = np.zeros(n_layers)

        for l in range(n_layers):
            db_index[l] = davies_bouldin_score(activities.loc[:, (l,)].fillna(0).to_numpy(), activities.loc[:, (l,)].index.get_level_values("Class"))

        figure.add_trace(
            go.Scatter(
                x=activities.columns.unique(level="Layer"), 
                y=db_index,
                name=label,
                mode="lines+markers"
            )
        )
    

    figure.update_xaxes(
        title_text="$l$"
    )
    figure.update_yaxes(
        title_text="$DB index$"
    )
    figure.update_layout(
        height=1000, 
        width=800, 
        title_text=title
    )

    return set_default_layout(figure)


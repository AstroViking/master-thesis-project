from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import davies_bouldin_score

from .figure import Figure


class DaviesBouldinIndex(Figure):
    def __init__(
        self, title: str, db_index_dict: Dict[str, pd.DataFrame], show_error_bars: bool = True
    ):
        super().__init__()

        for label, db_index in db_index_dict.items():

            layer_indices = db_index.columns.unique(level="Layer")

            self.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=db_index.loc[(slice(None), "DB Index")],
                    name=label,
                    error_y=dict(
                        type="data", array=db_index.loc[(slice(None), "Error")], visible=True
                    )
                    if show_error_bars
                    else None,
                    mode="lines+markers",
                    legendgroup="same",
                )
            )

        self.update_xaxes(title_text="$l$")
        self.update_yaxes(title_text="DB index")
        self.update_layout(height=1000, width=800, title_text=title)

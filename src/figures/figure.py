from abc import ABC
from pathlib import Path

import plotly.graph_objects as go


class Figure(go.Figure):
    def __init__(self):
        super().__init__()

        self.update_layout(
            template="plotly_white",
            font_family="STIX Two Text",
        )

    def save(self, output_path: Path):
        self.write_image(output_path, scale=3)

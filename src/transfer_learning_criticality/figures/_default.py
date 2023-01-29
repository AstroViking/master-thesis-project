import plotly.graph_objects as go

def set_default_layout(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        font_family="STIX Two Text",
    )

    return figure
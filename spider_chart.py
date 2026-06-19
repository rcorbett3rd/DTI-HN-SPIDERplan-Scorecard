from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_spider_chart(domain_df: pd.DataFrame):
    if domain_df is None or domain_df.empty:
        return None
    labels = domain_df["domain"].astype(str).tolist()
    values = domain_df["domain_score"].astype(float).tolist()
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_closed, theta=labels_closed, fill="toself", name="SPIDERPlan"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

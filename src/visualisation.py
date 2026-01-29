
import skforecast
import pandas as pd
import feature_engine
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
import nbformat


sales_history_all = (
    sales_history
    .groupby("week_id")
    .agg({"units_sold": 'sum'})
    .reset_index()
)


fig = go.Figure()
fig.add_trace(go.Scatter(x=data_train.index, y=data_train['units_sold'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=data_val.index, y=data_val['units_sold'], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=data_test.index, y=data_test['units_sold'], mode='lines', name='Test'))
fig.update_layout(
    title  = 'Units Sold',
    xaxis_title="date",
    yaxis_title="Units Sold",
    legend_title="Partition:",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1, xanchor="left", x=0.001)
)
#fig.update_xaxes(rangeslider_visible=True)
fig.show()
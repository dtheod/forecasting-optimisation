import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Forecasting & Optimization Dashboard")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

STATIC_COLS = [
    "store_id", "sku_id", "date", "units_sold",
    "category", "supplier_id", "lead_time_weeks", "case_pack",
    "moq", "unit_cost", "sell_price",
    "holding_cost_per_unit_week", "spoilage_rate_per_week",
    "capacity_units", "region",
]


@st.cache(allow_output_mutation=True)
def load_forecasts() -> pd.DataFrame:
    path = os.path.join("outputs", "forecasts.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return (
        pd.read_csv(path)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .assign(store_id=lambda d: d["item_id"].str.split("_").str[0])
        .assign(sku_id=lambda d: d["item_id"].str.split("_").str[1])
    )


@st.cache(allow_output_mutation=True)
def load_historical() -> pd.DataFrame:
    path = os.path.join("data", "processed", "processed_data.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    available = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in STATIC_COLS if c in available]
    return (
        pd.read_csv(path, usecols=use)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
    )


@st.cache(allow_output_mutation=True)
def load_optimization() -> pd.DataFrame:
    path = os.path.join("outputs", "optimization_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path).assign(date=lambda d: pd.to_datetime(d["date"]))


forecasts_df = load_forecasts()
historical_df = load_historical()
optimization_df = load_optimization()

if forecasts_df.empty:
    st.error("No forecast data found in outputs/forecasts.csv")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — Store / SKU selectors + Item metadata (Item 14)
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

stores = sorted(forecasts_df["store_id"].unique())
selected_store = st.sidebar.selectbox("Select Store", stores)

skus = sorted(
    forecasts_df.loc[forecasts_df["store_id"] == selected_store, "sku_id"].unique()
)
selected_sku = st.sidebar.selectbox("Select SKU", skus)
item_id = f"{selected_store}_{selected_sku}"

# --- Item 14: Sidebar metadata ---
if not historical_df.empty:
    item_meta = historical_df[
        (historical_df["store_id"] == selected_store)
        & (historical_df["sku_id"] == selected_sku)
    ]
    if not item_meta.empty:
        meta = item_meta.iloc[0]
        st.sidebar.markdown("---")
        st.sidebar.subheader("Item Details")
        detail_cols = {
            "category": "Category",
            "supplier_id": "Supplier",
            "lead_time_weeks": "Lead Time (weeks)",
            "case_pack": "Case Pack",
            "moq": "MOQ",
            "sell_price": "Sell Price",
            "unit_cost": "Unit Cost",
            "holding_cost_per_unit_week": "Holding Cost / unit / week",
            "spoilage_rate_per_week": "Spoilage Rate / week",
            "region": "Region",
        }
        for col, label in detail_cols.items():
            if col in meta.index and pd.notna(meta[col]):
                val = meta[col]
                if isinstance(val, float):
                    val = f"{val:,.4f}" if val < 1 else f"{val:,.2f}"
                st.sidebar.text(f"{label}: {val}")

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

fc = (
    forecasts_df[forecasts_df["item_id"] == item_id]
    .sort_values("date")
    .reset_index(drop=True)
)

hist = pd.DataFrame()
if not historical_df.empty:
    hist = (
        historical_df[
            (historical_df["store_id"] == selected_store)
            & (historical_df["sku_id"] == selected_sku)
        ]
        .sort_values("date")
        .tail(20)
        .reset_index(drop=True)
    )

opt = pd.DataFrame()
if not optimization_df.empty:
    opt = (
        optimization_df[
            (optimization_df["store_id"] == selected_store)
            & (optimization_df["item_id"] == item_id)
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title(f"Forecast & Optimization — {item_id}")

# ---- Forecast plot (Items 3 & 4) -----------------------------------------
st.subheader("Forecast vs Actuals")

fig = go.Figure()

# Prepare date / value lists
history_dates = list(hist["date"]) if not hist.empty else []
history_vals = list(hist["units_sold"]) if not hist.empty else []
fc_dates = list(fc["date"]) if not fc.empty else []

# 1) Historical actuals (black, like evaluation.py)
if history_dates:
    fig.add_trace(
        go.Scatter(
            x=history_dates,
            y=history_vals,
            mode="lines+markers",
            name="Actuals (units_sold)",
            line=dict(color="black"),
            marker=dict(size=4),
        )
    )

# Item 3: Bridge — connect last historical point to first forecast point
if history_dates and fc_dates:
    bridge_x = [history_dates[-1], fc_dates[0]]
    bridge_y_mean = [history_vals[-1], list(fc["mean_prediction"])[0]]
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y_mean,
            mode="lines",
            line=dict(color="green", dash="dot"),
            showlegend=False,
            name="bridge",
        )
    )

# 2) Confidence band p10–p90  (green shaded)
if not fc.empty and {"p10", "p90"}.issubset(fc.columns):
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=list(fc["p90"]),
            mode="lines", line=dict(width=0),
            showlegend=False, name="p90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=list(fc["p10"]),
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0,128,0,0.15)",
            name="CI (p10–p90)",
        )
    )

# 3) Forecast mean (green)
if not fc.empty and "mean_prediction" in fc.columns:
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=list(fc["mean_prediction"]),
            mode="lines",
            name="Forecast (mean)",
            line=dict(color="green"),
        )
    )

# Item 4: p50 median (dashed green)
if not fc.empty and "p50" in fc.columns:
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=list(fc["p50"]),
            mode="lines",
            name="Forecast (p50 median)",
            line=dict(color="green", dash="dash"),
        )
    )

fig.update_layout(
    xaxis_title="Date", yaxis_title="Units Sold",
    hovermode="x unified", legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Optimization section (Items 7 & 8) ----------------------------------
st.subheader("Optimization Results")

if not opt.empty:
    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Ordered Units", f"{opt['order_units'].sum():,.0f}")
    c2.metric("Total Lost Sales", f"{opt['lost_sales'].sum():,.2f}" if "lost_sales" in opt.columns else "N/A")
    c3.metric("Avg Projected Inventory", f"{opt['projected_inventory'].mean():,.2f}" if "projected_inventory" in opt.columns else "N/A")

    # --- Item 7: Multi-line chart ---
    opt_fig = go.Figure()
    opt_dates = list(opt["date"])

    if "order_units" in opt.columns:
        opt_fig.add_trace(go.Bar(
            x=opt_dates, y=list(opt["order_units"]),
            name="Order Units", marker_color="steelblue", opacity=0.6,
        ))
    if "projected_inventory" in opt.columns:
        opt_fig.add_trace(go.Scatter(
            x=opt_dates, y=list(opt["projected_inventory"]),
            mode="lines+markers", name="Projected Inventory",
            line=dict(color="orange"),
        ))
    if "lost_sales" in opt.columns:
        opt_fig.add_trace(go.Scatter(
            x=opt_dates, y=list(opt["lost_sales"]),
            mode="lines+markers", name="Lost Sales",
            line=dict(color="red"),
        ))
    if "forecast_mean" in opt.columns:
        opt_fig.add_trace(go.Scatter(
            x=opt_dates, y=list(opt["forecast_mean"]),
            mode="lines", name="Forecast Demand",
            line=dict(color="green", dash="dot"),
        ))

    opt_fig.update_layout(
        title="Weekly Optimization Breakdown",
        xaxis_title="Date", yaxis_title="Units",
        hovermode="x unified", barmode="overlay",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(opt_fig, use_container_width=True)

    # --- Item 8: Cost breakdown pie chart ---
    # Retrieve item-level cost parameters from historical data
    if not hist.empty:
        m = hist.iloc[0]
        uc = float(m.get("unit_cost", 0) or 0)
        hc = float(m.get("holding_cost_per_unit_week", 0) or 0)
        sp = float(m.get("spoilage_rate_per_week", 0) or 0)
        price = float(m.get("sell_price", 0) or 0)
        stockout_penalty = price * 1.5 if price > 0 else uc * 2

        purchase_cost = opt["order_units"].sum() * uc
        holding_cost = opt["projected_inventory"].sum() * hc if "projected_inventory" in opt.columns else 0
        spoilage_cost = opt["projected_inventory"].sum() * sp * uc if "projected_inventory" in opt.columns else 0
        stockout_cost = opt["lost_sales"].sum() * stockout_penalty if "lost_sales" in opt.columns else 0

        cost_labels = ["Purchase", "Holding", "Spoilage", "Stockout Penalty"]
        cost_values = [purchase_cost, holding_cost, spoilage_cost, stockout_cost]

        # Only show if there are non-zero costs
        if any(v > 0 for v in cost_values):
            cost_fig = go.Figure(
                go.Pie(
                    labels=cost_labels,
                    values=cost_values,
                    hole=0.4,
                    marker_colors=["#636EFA", "#EF553B", "#FFA15A", "#AB63FA"],
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value:,.2f}<extra></extra>",
                )
            )
            cost_fig.update_layout(title="Cost Breakdown")
            st.plotly_chart(cost_fig, use_container_width=True)

    # Data table
    show_cols = [c for c in ["date", "order_units", "projected_inventory", "lost_sales", "forecast_mean"] if c in opt.columns]
    st.dataframe(opt[show_cols])
else:
    st.info("No optimization results for this selection.")

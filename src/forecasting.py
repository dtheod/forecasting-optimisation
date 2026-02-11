import os
from pathlib import Path
import functools
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

if hasattr(torch, 'load'):
    _original_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from prefect import task

@task
def build_listdataset(df, target_col, timestamp_col, item_id_col,
                      feat_dynamic_real_cols, feat_static_cat_cols, feat_static_real_cols, freq):
    data_list = []
    for series_id, group in df.groupby(item_id_col):
        group = group.sort_values(timestamp_col)

        target = group[target_col].to_numpy(dtype=np.float32)
        start = pd.to_datetime(group[timestamp_col].iloc[0])

        entry = {"start": start, "target": target, "item_id": series_id}

        if feat_dynamic_real_cols:
            entry["feat_dynamic_real"] = group[feat_dynamic_real_cols].to_numpy(dtype=np.float32).T

        if feat_static_cat_cols:
            entry["feat_static_cat"] = group[feat_static_cat_cols].iloc[0].to_numpy(dtype=np.int64)

        if feat_static_real_cols:
            entry["feat_static_real"] = group[feat_static_real_cols].iloc[0].to_numpy(dtype=np.float32)

        data_list.append(entry)

    return ListDataset(data_list, freq=freq)

@task
def get_feature_configuration(df):
    """
    Extracts feature columns and cardinality from dataframe.
    """
    feat_dynamic_real_cols = [
        "sell_price", "promo_flag", "price_multiplier",
        "stockout_flag", "holiday_flag", "temp_index",
        "month_sin", "month_cos", "woy_sin", "woy_cos",
        "units_sold_lag_1", "units_sold_lag_4", "units_sold_lag_12",
        "units_sold_window_4_mean", "units_sold_window_4_std",
        "units_sold_window_12_mean", "units_sold_window_12_std"
    ]
    feat_dynamic_real_cols = [c for c in feat_dynamic_real_cols if c in df.columns]

    feat_static_cat_cols = []
    cardinality = None
    if "supplier_id" in df.columns:
        # Assumes df has been cleaned or we detect what's there
        # We DO NOT mutate df here to avoid side effects if called on same df.
        # But we need cardinality from the data.
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df["supplier_id"]):
             feat_static_cat_cols = ["supplier_id"]
             cardinality = [int(df["supplier_id"].max() + 1)] # Use max+1 or nunique if dense
             # In train_model_forecasting we used nunique() but max+1 is safer for embeddings if IDs are indices
             # Original code: cardinality = [int(df["supplier_id"].nunique())]
             cardinality = [int(df["supplier_id"].nunique())]
    
    feat_static_real_cols = ["lead_time_weeks", "moq", "unit_cost", "spoilage_rate_per_week", "max_weekly_supply_units"]
    feat_static_real_cols = [c for c in feat_static_real_cols if c in df.columns]
    
    
    return feat_dynamic_real_cols, feat_static_cat_cols, feat_static_real_cols, cardinality

@task
def create_gluonts_dataset_from_df(df, freq="W-MON"):
    """
    Creates a GluonTS ListDataset from dataframe for tuning/evaluation,
    ensuring consistent preprocessing.
    """
    df = df.copy() # Safe copy
    
    if "supplier_id" in df.columns:
        df["supplier_id"] = pd.to_numeric(df["supplier_id"], errors="coerce").fillna(0).astype(int)
        df.loc[df["supplier_id"] < 0, "supplier_id"] = 0

    (feat_dynamic_real_cols, feat_static_cat_cols, 
     feat_static_real_cols, cardinality) = get_feature_configuration(df)
     
    target_col = "units_sold"
    timestamp_col = "date"
    item_id_col = "series_id"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    return build_listdataset.fn(
        df, target_col, timestamp_col, item_id_col,
        feat_dynamic_real_cols, feat_static_cat_cols, feat_static_real_cols, freq
    )


@task
def train_model_forecasting(df: pd.DataFrame, config, model_dir="models/gluonts_model"):
    prediction_length = config.model.horizon

    df["date"] = pd.to_datetime(df["date"])

    #Define the columns needed for GluonTS
    target_col = "units_sold"
    timestamp_col = "date"
    item_id_col = "series_id"
    freq = "W-MON"

    (feat_dynamic_real_cols, feat_static_cat_cols, 
     feat_static_real_cols, cardinality) = get_feature_configuration(df)

    # ---- Proper train/test split: hold out last horizon per series
    def truncate_last_h(g):
        return g.iloc[:-prediction_length] if len(g) > prediction_length else g.iloc[:0]

    train_df = df.groupby(item_id_col, group_keys=False).apply(truncate_last_h)
    test_df = df  # full series for evaluation

    train_ds = build_listdataset(train_df, target_col, timestamp_col, item_id_col,
                                 feat_dynamic_real_cols, feat_static_cat_cols, feat_static_real_cols, freq)
    test_ds = build_listdataset(test_df, target_col, timestamp_col, item_id_col,
                                feat_dynamic_real_cols, feat_static_cat_cols, feat_static_real_cols, freq)

    # Get hyperparameters with defaults
    context_length = config.model.get("context_length", prediction_length)
    epochs = config.model.get("epochs", 50)
    num_layers = config.model.get("num_layers", 2)
    hidden_size = config.model.get("hidden_size", 40)
    dropout_rate = config.model.get("dropout", 0.1)
    
    print(f"Hyperparameters: Context={context_length}, Epochs={epochs}, Layers={num_layers}, Hidden={hidden_size}, Dropout={dropout_rate}")

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
        trainer_kwargs={
            "max_epochs": epochs, 
            "accelerator": "cpu", 
            "devices": 1
        },
        distr_output=NegativeBinomialOutput(),
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate, 
        num_feat_dynamic_real=len(feat_dynamic_real_cols),
        num_feat_static_cat=len(feat_static_cat_cols),
        num_feat_static_real=len(feat_static_real_cols),
        cardinality=cardinality
    )

    predictor = estimator.train(train_ds, num_workers=0)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=200
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("Evaluation Metrics:")
    # print("Available Metrics:", list(agg_metrics.keys()))
    print("mean_wQuantileLoss:", agg_metrics.get("mean_wQuantileLoss"))
    
    # Calculate MAE manually if not present
    if "MAE" in agg_metrics:
        print("MAE:", agg_metrics["MAE"])
    elif "abs_error" in agg_metrics:
        # total_prediction_length is often num_series * prediction_length
        # But let's check if we have a count or calculate it from item_metrics
        # item_metrics has 'abs_error' per item, and we know prediction_length
        num_obs = len(item_metrics) * prediction_length
        calc_mae = agg_metrics["abs_error"] / num_obs if num_obs > 0 else 0.0
        print(f"MAE (Calculated): {calc_mae}")
        agg_metrics["MAE"] = calc_mae
    else:
        print("MAE: Not available")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    predictor.serialize(Path(model_dir))
    print(f"Model saved to {model_dir}")

    return agg_metrics
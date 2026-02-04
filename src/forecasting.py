import os
from pathlib import Path
import functools
import numpy as np
import pandas as pd
import torch
import warnings


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
def train_model_forecasting(df: pd.DataFrame, config, model_dir="models/gluonts_model"):
    prediction_length = config.model.horizon

    df["date"] = pd.to_datetime(df["date"])

    target_col = "units_sold"
    timestamp_col = "date"
    item_id_col = "series_id"
    freq = "W-MON"

    feat_dynamic_real_cols = [
        "sell_price", "promo_flag", "price_multiplier",
        "stockout_flag", "holiday_flag", "temp_index",
        "month_sin", "month_cos", "woy_sin", "woy_cos"
    ]
    feat_dynamic_real_cols = [c for c in feat_dynamic_real_cols if c in df.columns]

    feat_static_cat_cols = []
    cardinality = None
    if "supplier_id" in df.columns:
        # Ensure non-negative ints
        df["supplier_id"] = pd.to_numeric(df["supplier_id"], errors="coerce").fillna(0).astype(int)
        df.loc[df["supplier_id"] < 0, "supplier_id"] = 0

        feat_static_cat_cols = ["supplier_id"]
        cardinality = [int(df["supplier_id"].nunique())]  # safer than max+1

    feat_static_real_cols = ["lead_time_weeks", "moq", "unit_cost", "spoilage_rate_per_week", "max_weekly_supply_units"]
    feat_static_real_cols = [c for c in feat_static_real_cols if c in df.columns]

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
    epochs = config.model.get("epochs", 20)
    num_layers = config.model.get("num_layers", 2)
    hidden_size = config.model.get("hidden_size", 40)
    
    print(f"Hyperparameters: Context={context_length}, Epochs={epochs}, Layers={num_layers}, Hidden={hidden_size}")

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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("Evaluation Metrics:")
    print("mean_wQuantileLoss:", agg_metrics.get("mean_wQuantileLoss"))
    print("RMSE:", agg_metrics.get("RMSE"))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    predictor.serialize(Path(model_dir))
    print(f"Model saved to {model_dir}")

    return agg_metrics
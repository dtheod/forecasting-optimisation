import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.evaluation import make_evaluation_predictions
from forecasting import build_listdataset

def visualize_forecasts(features: pd.DataFrame, 
                        config,
                        num_samples=5, 
                        output_dir="outputs/plots"):
    """
    Visualizes the last 4 weeks of sales predictions vs actuals.
    """
    print(f"\n[EVALUATION] Visualizing Forecasts...")
    
    model_dir = config.model.dir
    prediction_length = config.model.horizon
    freq = "W-MON" # Hardcoded consistent with forecasting.py
    
    # 1. Load Model
    # Important: Re-apply monkey patch if needed here? 
    # Usually importing 'forecasting' handles it if it's at module level.
    # But best to be safe if running standalone.
    # The patch is in forecasting.py which is imported above.
    
    try:
        predictor = Predictor.deserialize(Path(model_dir))
    except Exception as e:
        print(f"Failed to load predictor from {model_dir}: {e}")
        return

    # 2. Prepare Data (Same as Testing Split)
    # We need the full dataset to plot actuals
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    target_col = "units_sold"
    timestamp_col = "date"
    item_id_col = "series_id"
    
    # Extract feature columns logic (duplicated from forecasting.py, ideally refactor to common config)
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
    if "supplier_id" in df.columns:
         # Ensure transform matches training
        df["supplier_id"] = pd.to_numeric(df["supplier_id"], errors="coerce").fillna(0).astype(int)
        df.loc[df["supplier_id"] < 0, "supplier_id"] = 0
        feat_static_cat_cols = ["supplier_id"]

    feat_static_real_cols = ["lead_time_weeks", "moq", "unit_cost", "spoilage_rate_per_week", "max_weekly_supply_units"]
    feat_static_real_cols = [c for c in feat_static_real_cols if c in df.columns]
    
    # Build dataset for plotting (full series)
    test_ds = build_listdataset.fn( # Accessing the underlying function of the Prefect task
        df, target_col, timestamp_col, item_id_col,
        feat_dynamic_real_cols, feat_static_cat_cols, feat_static_real_cols, freq
    )

    # 3. Generate Forecasts
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # 4. Plot Samples
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples or first N
    import random
    indices = range(len(forecasts))
    sample_indices = random.sample(indices, min(num_samples, len(indices)))
    
    for idx in sample_indices:
        forecast = forecasts[idx]
        ts = tss[idx]
        item_id = forecast.item_id
        
        # We only want to visualize slightly more than the prediction length locally
        # e.g. last 12 weeks + prediction
        
        plt.figure(figsize=(10, 5))
        
        # Plot the TimeSeries (Actuals)
        # Slicing the dataframe to last 20 points for clarity
        ts_tail = ts[-20:] 
        
        # Plot Actuals
        # Convert index to timestamp if it is PeriodIndex, otherwise matplotlib might fail
        plot_index = ts_tail.index.to_timestamp() if hasattr(ts_tail.index, 'to_timestamp') else ts_tail.index
        plt.plot(plot_index, ts_tail.values, label="Actuals", color='black', marker='.')
        
        # Plot Forecast
        forecast.plot(color='green')
        
        plt.title(f"Forecast vs Actuals: {item_id}")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        safe_name = item_id.replace("/", "_")
        plt.savefig(f"{output_dir}/forecast_{safe_name}.png")
        plt.close()
        
    print(f"Saved {len(sample_indices)} plots to {output_dir}")

    # 5. Export Forecasts to CSV
    print(f"[EVALUATION] Exporting forecasts to CSV...")
    export_data = []
    
    for forecast, ts in zip(forecasts, tss):
        # Determine the prediction interval
        start_date = forecast.start_date
        # Create a period range for the forecast horizon
        forecast_dates = pd.period_range(
             start=start_date, 
             periods=prediction_length, 
             freq=freq
        ).to_timestamp()
        
        # Get actuals for this range
        # ts is a pandas Series with PeriodIndex. Convert to timestamp for alignment if needed?
        # Let's slice ts by the forecast range.
        # ts index is PeriodIndex. forecast_dates is DatetimeIndex.
        # It's easier to work with Periods for slicing then convert.
        
        forecast_periods = pd.period_range(
             start=start_date, 
             periods=prediction_length, 
             freq=freq
        )
        
        actuals = ts.loc[forecast_periods].values if hasattr(ts, 'loc') else np.full(prediction_length, np.nan)
        # Handle case where indices might not match perfectly or missing data (fill NaN)
        # Robust fetch:
        actual_values = []
        for p in forecast_periods:
             try:
                 val = ts.loc[p]
                 if isinstance(val, (pd.Series, pd.DataFrame)):
                     val = val.iloc[0]
                 actual_values.append(val)
             except KeyError:
                 actual_values.append(None)
        
        # Predictions
        p10 = forecast.quantile(0.1)
        p50 = forecast.quantile(0.5)
        p90 = forecast.quantile(0.9)
        mean = forecast.mean
        
        for i, date in enumerate(forecast_dates):
            export_data.append({
                "item_id": forecast.item_id,
                "date": date,
                "actual": actual_values[i],
                "mean_prediction": mean[i],
                "p10": p10[i],
                "p50": p50[i],
                "p90": p90[i]
            })
            
    export_df = pd.DataFrame(export_data)
    output_csv = f"{output_dir}/../forecasts.csv" # Save in outputs/ root
    # Or just outputs/forecasts.csv if output_dir is outputs/plots
    # Let's put it in outputs/forecasts.csv
    
    # Normalize output path
    base_output_dir = os.path.dirname(output_dir) # e.g. "outputs"
    csv_path = os.path.join(base_output_dir, "forecasts.csv")
    
    export_df.to_csv(csv_path, index=False)
    print(f"Saved forecast data to {csv_path}")

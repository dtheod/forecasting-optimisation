"""Feature engineering script"""
import pandas as pd
from typing import Tuple, Dict
from omegaconf import DictConfig
from prefect import task
import numpy as np
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures

@task
def prepare_data_for_training(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for GluonTS training:
    1. Creates a complete grid (Series X Date) from min to max date.
    2. Merges processed data onto the grid.
    3. Fills missing values (Targets=0, Static=FFill, Dynamic=FFill/BFill).
    4. Computes calendar features.
    5. Returns specific columns requested.
    """
    
    # 1. Prepare base data
    df = (
        processed_data
        .assign(
            date = lambda df_: pd.to_datetime(df_['date']),
            series_id = lambda df_: df_['store_id'] + '_' + df_['sku_id']
        )
        .assign(series_id = lambda df_: df_['series_id'].astype(str))
    )
    
    # 2. Create the initial dataframe with all series and dates
    all_series = df["series_id"].unique()
    date_range = pd.date_range(start=df["date"].min(),
                               end=df["date"].max(),
                               freq="W-MON"
                               )
    
    # 3. Create the full grid using MultiIndex
    full_idx = pd.MultiIndex.from_product([date_range, all_series], names=["date", "series_id"])
    
    cols_to_fill = [
        "sell_price", "promo_flag", "price_multiplier",
        "stockout_flag", "holiday_flag",
        "temp_index", "supplier_id", "lead_time_weeks",
        "moq", "unit_cost", "spoilage_rate_per_week",
        "max_weekly_supply_units"
    ]
    
    full_df = (
        pd.DataFrame(index=full_idx)
        .reset_index()
        .merge(df, on=["date", "series_id"], how="left")
        .assign(units_sold=lambda df_: df_["units_sold"].fillna(0.0))
        .sort_values(["series_id", "date"])
        .assign(**{
            c: (lambda col: (lambda d, col=col: d.groupby("series_id")[col].ffill().bfill()))(c)
            for c in cols_to_fill
        })
    )
           
    num_cols = full_df[cols_to_fill].select_dtypes(include=[np.number]).columns
    full_df[num_cols] = full_df[num_cols].fillna(0)
    
    # Compute Calendar Features
    features = (
        full_df
        .assign(
            week_of_year = lambda df_: df_['date'].dt.isocalendar().week.astype(int),
            month_sin = lambda df_: np.sin(2 * np.pi * df_["month"]/12),
            month_cos = lambda df_: np.cos(2 * np.pi * df_["month"]/12),
            woy_sin = lambda df_: np.sin(2 * np.pi * df_["week_of_year"]/52.18),
            woy_cos = lambda df_: np.cos(2 * np.pi * df_["week_of_year"]/52.18)
        )
        .astype(
            {
                "supplier_id": "category",
                "season": "category",
                "region": "category",
                "category": "category"
            }
        )
        .drop(["store_id", "sku_id"], axis = 1)
    )

    # 8b. Add Lag and Rolling Features using feature_engine
    # We need to apply these PER series_id.
    
    # Lag Features: Lag 1, 4 (month), 12 (quarter)
    lag_transformer = LagFeatures(
        variables=["units_sold"],
        periods=[1, 4, 12],
        missing_values="ignore",
        fill_value=0
    )
    
    # Window Features: Rolling Mean/Std for 4 and 12 weeks
    window_transformer = WindowFeatures(
        variables=["units_sold"],
        window=[4, 12],
        functions=["mean", "std"],
        missing_values="ignore"
    )
    
    print("Generating Lag and Window features...")
    
    def apply_ts_transform(df_group):
        # Local transform
        df_group = lag_transformer.fit_transform(df_group)
        df_group = window_transformer.fit_transform(df_group)
        return df_group

    features = features.groupby("series_id", group_keys=False).apply(apply_ts_transform)
    
    # Fill resultant NaNs from lags (e.g. first few rows) with 0
    new_features = [
        "units_sold_lag_1", "units_sold_lag_4", "units_sold_lag_12",
        "units_sold_window_4_mean", "units_sold_window_4_std",
        "units_sold_window_12_mean", "units_sold_window_12_std"
    ]

    features = (
        features
        .fillna({
            "units_sold_window_4_mean": 0,
            "units_sold_window_4_std": 0,
            "units_sold_window_12_mean": 0,
            "units_sold_window_12_std": 0
        })
    )
            
    return features
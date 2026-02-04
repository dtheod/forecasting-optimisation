import pandas as pd
from typing import Tuple, Dict
from omegaconf import DictConfig
from prefect import task
import numpy as np

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
        processed_data.copy()
        .assign(
            date = lambda df_: pd.to_datetime(df_['date']),
            series_id = lambda df_: df_['store_id'] + '_' + df_['sku_id']
        )
        .assign(series_id = lambda df_: df_['series_id'].astype(str))
    )
    
    # 2. Extract unique series_ids and full date range
    all_series = df["series_id"].unique()
    date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="W-MON")
    
    # 3. Create Full Grid using MultiIndex
    full_idx = pd.MultiIndex.from_product([date_range, all_series], names=["date", "series_id"])
    full_df = pd.DataFrame(index=full_idx).reset_index()
    
    # 4. Merge Data
    full_df = full_df.merge(df, on=["date", "series_id"], how="left")
    
    # 5. Handle Target (units_sold) - Fill NaNs with 0
    full_df["units_sold"] = full_df["units_sold"].fillna(0.0)
    

    full_df = full_df.sort_values(["series_id", "date"])
    
    cols_to_fill = [
        "sell_price", "promo_flag", "price_multiplier", "stockout_flag", "holiday_flag", "temp_index", # Dynamic
        "supplier_id", "lead_time_weeks", "moq", "unit_cost", "spoilage_rate_per_week", "max_weekly_supply_units" # Static
    ]
    
    # We only fill columns that exist
    cols_to_fill = [c for c in cols_to_fill if c in full_df.columns]
    
    # GroupBy series_id to prevent bleeding between series, then ffill/bfill
    # Note: Vectorized ffill/bfill on the whole DF sorted by series_id is faster, 
    # but strictly we should mask series changes. 
    # However, pandas groupby.ffill is robust.
    
    full_df[cols_to_fill] = full_df.groupby("series_id")[cols_to_fill].ffill().bfill()
    
    # Safety fill for any remaining NaNs (e.g. if a series has NO data for a column at all)
    # Numeric -> 0, Categorical/Object -> Unknown or 0?
    # Let's fill remaining numeric with 0.
    num_cols = full_df[cols_to_fill].select_dtypes(include=[np.number]).columns
    full_df[num_cols] = full_df[num_cols].fillna(0)
    
    # 7. Compute Calendar Features
    full_df["month"] = full_df["date"].dt.month
    full_df["week_of_year"] = full_df["date"].dt.isocalendar().week.astype(int)
    
    full_df["month_sin"] = np.sin(2*np.pi*full_df["month"]/12)
    full_df["month_cos"] = np.cos(2*np.pi*full_df["month"]/12)
    full_df["woy_sin"] = np.sin(2*np.pi*full_df["week_of_year"]/52.18)
    full_df["woy_cos"] = np.cos(2*np.pi*full_df["week_of_year"]/52.18)
    
    # 8. Encode Supplier ID if needed (for GluonTS we might want INT codes)
    if "supplier_id" in full_df.columns:
        full_df["supplier_id"] = full_df["supplier_id"].astype("category").cat.codes
        
    # 9. Select Final Columns
    final_cols = [
        'date', 'series_id', 'units_sold', 
        'month', 'week_of_year', 'month_sin', 'month_cos', 'woy_sin', 'woy_cos', 
        'sell_price', 'promo_flag', 'price_multiplier', 'stockout_flag', 'holiday_flag', 'temp_index',
        'supplier_id', 'lead_time_weeks', 'moq', 'unit_cost', 
        'spoilage_rate_per_week', 'max_weekly_supply_units'
    ]
    
    # Ensure they exist (some might be missing from input data)
    final_cols = [c for c in final_cols if c in full_df.columns]
    features = full_df[final_cols]

    return features
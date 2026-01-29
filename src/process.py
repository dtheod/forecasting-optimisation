"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Danis Theodoulou
"""

import hydra
from omegaconf import DictConfig
import pandas as pd



def create_datetime(calendar: pd.DataFrame) -> pd.DataFrame:
    # Define start date (Monday)
    start_date = pd.Timestamp("2021-01-04")

    # Create date column (start of week)
    calendar["date"] = start_date + pd.to_timedelta(calendar["week_id"] * 7, unit="D")

    # Optional: ensure it's datetime64
    calendar["date"] = pd.to_datetime(calendar['date'], format='%Y-%m-%d %H:%M:%S')

    return calendar




@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    """Function to process the data"""
    calendar = (
        pd.read_csv(config.data.raw.calendar)
        .pipe(create_datetime)
    )
    sales_history = pd.read_csv(config.data.raw.sales_history)
    prices_promos = pd.read_csv(config.data.raw.prices_promos)
    stores = pd.read_csv(config.data.raw.stores)
    suppliers = pd.read_csv(config.data.raw.suppliers)
    initial_inventory = pd.read_csv(config.data.raw.initial_inventory)
    skus = pd.read_csv(config.data.raw.skus)

    processed_data = (
        sales_history
        .merge(calendar, how = "left", on = "week_id")
        .merge(stores, how = "left", on = "store_id")
        .merge(skus, how = "left", on = "sku_id")
        .merge(suppliers, how = "left", on = "supplier_id")
        .merge(prices_promos, how = "left", on = ["store_id", "sku_id", "week_id"])
    )

    print(processed_data.head())
    processed_data.to_csv(config.data.processed.processed_data, index = False)
    return None




if __name__ == "__main__":
    process_data()

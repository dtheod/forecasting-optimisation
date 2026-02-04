"""
Author: Danis Theodoulou
"""

import hydra
from prefect import task
from omegaconf import DictConfig
import pandas as pd

@task
def process_data(sales_history, calendar, stores, skus, suppliers, prices_promos):
    """Actual processing logic"""

    processed_data = (
        sales_history
        .merge(calendar, how = "left", on = "week_id")
        .merge(stores, how = "left", on = "store_id")
        .merge(skus, how = "left", on = "sku_id")
        .merge(suppliers, how = "left", on = "supplier_id")
        .merge(prices_promos, how = "left", on = ["store_id", "sku_id", "week_id"])
    )
    
    return processed_data

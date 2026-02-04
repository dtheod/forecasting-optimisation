
import hydra
from omegaconf import DictConfig
from process import process_data
from feature_engineering import prepare_data_for_training
from forecasting import train_model_forecasting
from utils import create_datetime
from prefect import flow
from datetime import datetime
from typing import Tuple, Dict
import pandas as pd

DATE = datetime.today().strftime("%Y-%m-%d %H:%M")

@flow(flow_run_name=f"ML training pipeline run on {DATE}", log_prints=True)
def training_pipeline(config: DictConfig) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    
    # 1. Read Data
    sales_history = pd.read_csv(config.data.raw.sales_history)
    prices_promos = pd.read_csv(config.data.raw.prices_promos)
    stores = pd.read_csv(config.data.raw.stores)
    suppliers = pd.read_csv(config.data.raw.suppliers)
    initial_inventory = pd.read_csv(config.data.raw.initial_inventory)
    skus = pd.read_csv(config.data.raw.skus)
    calendar = (
        pd.read_csv(config.data.raw.calendar)
        .pipe(create_datetime)
    )
    
    
    print("\n[MAIN] Starting Data Processing...")
    processed_data = process_data(sales_history, calendar, stores, skus, suppliers, prices_promos)
    
    processed_data.to_csv(config.data.processed.path, index=False)
    
    print("\n[MAIN] Starting Feature Engineering...")
    # Returns single long dataframe for GluonTS
    
    features = prepare_data_for_training(processed_data)

    features.to_csv(config.data.features.path, index=False)
 
    # # 2. Train Model
    print("\n[MAIN] Starting Model Training...")
    predictions = train_model_forecasting(df = features,
                                          config = config,
                                          model_dir = config.model.dir)

    # 3. Visualize

    from evaluation import visualize_forecasts
    visualize_forecasts(features, config, num_samples=10)

    return None #predictions


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:
    predictions = training_pipeline(config)



if __name__ == "__main__":
    main()

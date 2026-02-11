
import hydra
import warnings
# Suppress GluonTS/PyTorch indexing warning
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing is deprecated")
# Suppress Pandas masked element warning
warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")
from omegaconf import DictConfig
from process import process_data
from feature_engineering import prepare_data_for_training
from forecasting import train_model_forecasting, create_gluonts_dataset_from_df
from tuning import run_tuning
from utils import create_datetime
from prefect import flow
from datetime import datetime
from typing import Tuple, Dict
import pandas as pd
from evaluation import visualize_forecasts
from optimization import InventoryOptimizer

DATE = datetime.today().strftime("%Y-%m-%d %H:%M")

@flow(flow_run_name=f"ML training pipeline run on {DATE}", log_prints=True)
def training_evaluation_pipeline(config: DictConfig) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    
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

    # features.to_csv(config.data.features.path, index=False)
 
    # # 2. Hyperparameter Tuning (Optional)
    # if config.tuning.get("enabled", False):
    #     print("\n[MAIN] Starting Hyperparameter Tuning...")
    #     dataset = create_gluonts_dataset_from_df(features)
        
    #     best_params = run_tuning(
    #         dataset, 
    #         prediction_length=config.model.horizon, 
    #         freq="W-MON", 
    #         n_trials=config.tuning.n_trials
    #     )
        
    #     print(f"\n[MAIN] Tuning Complete. Best Params: {best_params}")
        
    #     # Override config with best params
    #     config.model.context_length = best_params.get("context_length", config.model.context_length)
    #     config.model.num_layers = best_params.get("num_layers", config.model.num_layers)
    #     config.model.hidden_size = best_params.get("hidden_size", config.model.hidden_size)
    #     config.model.epochs = config.model.epochs
    #     config.model.dropout = best_params.get("dropout_rate", config.model.get("dropout", 0.1))

    # # 3. Train Model
    # print("\n[MAIN] Starting Model Training...")
    # predictions = train_model_forecasting(df = features,
    #                                       config = config,
    #                                       model_dir = config.model.dir)

    # # 3. Visualize and Export Forecasts
    # visualize_forecasts(features, config, num_samples=100)
    
    # 4. Inventory Optimization
    if config.get("optimization", {}).get("enabled", True): # Enable by default for now
        print("\n[MAIN] Starting Inventory Optimization...")
        try:
            # Read back the forecasts we just exported
            forecasts_path = "outputs/forecasts.csv"
            forecast_df = pd.read_csv(forecasts_path)
            
            optimizer = InventoryOptimizer(forecast_df, features)
            optimization_results = optimizer.optimize_all(horizon_weeks=config.model.horizon)
            
            if not optimization_results.empty:
                opt_path = "outputs/optimization_results.csv"
                optimization_results.to_csv(opt_path, index=False)
                print(f"[MAIN] Optimization results saved to {opt_path}")
            else:
                print("[MAIN] Optimization returned no results.")
                
        except Exception as e:
            print(f"[MAIN] Optimization failed: {e}")
            import traceback
            traceback.print_exc()

    return None #predictions


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:
    predictions = training_evaluation_pipeline(config)



if __name__ == "__main__":
    main()

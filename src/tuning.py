import optuna
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.split import split
from gluonts.evaluation import Evaluator
from gluonts.torch.distributions import NegativeBinomialOutput

class DeepARTuningObjective:
    def __init__(
        self, dataset, prediction_length, freq, metric_type="mean_wQuantileLoss"
    ):
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.freq = freq
        self.metric_type = metric_type

        self.train, self.test_template = split(dataset, offset=-self.prediction_length)
        
        # Generate validation instances (input, label)
        self.validation = list(self.test_template.generate_instances(
            prediction_length=prediction_length
        ))
        
        # Predictor expects list of dicts (input)
        self.validation_input = [entry[0] for entry in self.validation]
        
        # Evaluator expects list of pandas Series (ground truth) for 'tss'
        # entry[1] is the label data entry. We need to convert it to pandas Series.
        from gluonts.dataset.common import ListDataset
        # Helper to convert data entry to pandas series with correct index
        def to_series(entry):
            index = pd.period_range(
                start=entry["start"],
                periods=len(entry["target"]),
                freq=entry["start"].freq
            )
            return pd.Series(entry["target"], index=index)

        self.validation_label = [to_series(entry[1]) for entry in self.validation]

    def get_params(self, trial) -> dict:
        return {
            "context_length": trial.suggest_int("context_length", 25, 52), # Reduced max for speed
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "hidden_size": trial.suggest_int("hidden_size", 20, 60),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),
            "epochs": 50 # FAST tuning
        }

    def __call__(self, trial):
        params = self.get_params(trial)
        
        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=params["context_length"],
            freq=self.freq,
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": params["epochs"],
                "accelerator": "cpu",
                "devices": 1
            },
            distr_output=NegativeBinomialOutput(),
            num_layers=params["num_layers"],
            hidden_size=params["hidden_size"],
            dropout_rate=params["dropout_rate"]
        )

        predictor = estimator.train(self.train, num_workers=0)
        
        forecast_it = predictor.predict(self.validation_input)
        forecasts = list(forecast_it)

        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        # validation_label is list of Series (tss), forecasts is list of Forecast
        agg_metrics, item_metrics = evaluator(
            self.validation_label, 
            forecasts
        )
        return agg_metrics[self.metric_type]

def run_tuning(dataset, prediction_length, freq, n_trials=5):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        DeepARTuningObjective(dataset, prediction_length, freq),
        n_trials=n_trials,
    )
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    # Generate and save plots
    try:
        import os
        from optuna.visualization import (
            plot_contour,
            plot_edf,
            plot_intermediate_values,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_rank,
            plot_slice,
            plot_timeline
        )
        
        output_dir = "outputs/optuna_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        plots = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            "contour": plot_contour,
            "edf": plot_edf,
            "intermediate_values": plot_intermediate_values,
            "parallel_coordinate": plot_parallel_coordinate,
            "rank": plot_rank,
            "slice": plot_slice,
            "timeline": plot_timeline
        }

        print(f"\n[TUNING] Saving Optuna plots to {output_dir}...")
        
        for name, plot_func in plots.items():
            try:
                # Some plots might fail if not enough data or specific prerequisites (like intermediate values) are missing
                fig = plot_func(study)
                # Save as PNG (static image)
                fig.write_image(f"{output_dir}/{name}.png")
                # print(f"  Saved {name}.png")
            except Exception as e:
                print(f"  Could not generate {name} plot: {e}")
                
        print("[TUNING] Plots saved (PNG).")
                
    except ImportError:
        print("Optuna visualization dependencies (plotly, kaleido) not installed. Skipping plots.")
    except Exception as e:
        print(f"Error saving tuning plots: {e}")
        
    return trial.params

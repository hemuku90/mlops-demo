import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.onnx
import sys
import os
import argparse
import dagshub
import yaml
import itertools
import shutil
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def get_model(algo_name, params, seed):
    if algo_name == "elastic_net":
        return ElasticNet(
            alpha=params["alpha"], 
            l1_ratio=params["l1_ratio"], 
            random_state=seed
        )
    elif algo_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"], 
            max_depth=params["max_depth"], 
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def train_optimization():
    # Initialize Dagshub
    dagshub.init(repo_owner='hemantku1990', repo_name='my-first-repo', mlflow=True)
    
    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    seed = params.get("seed", 42)
    experiment_name = params.get("experiment_name", "Default_Experiment")
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load data
    csv_url = os.path.join("data", "wine_quality.csv")
    try:
        data = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Unable to read file. Error: {e}")
        return

    # Split data
    train, test = train_test_split(data, random_state=seed)

    train_x = train.drop(["target"], axis=1)
    test_x = test.drop(["target"], axis=1)
    train_y = train[["target"]]
    test_y = test[["target"]]

    # MLflow tracking
    print(f"Logging to MLflow at {mlflow.get_tracking_uri()}")

    best_run_id = None
    best_rmse = float("inf")
    best_model = None
    best_algo_name = ""

    # Iterate over enabled algorithms
    for algo_name in params["enabled_algorithms"]:
        search_space = params["search_space"][algo_name]
        
        # Generate parameter grid
        keys, values = zip(*search_space.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for run_params in param_combinations:
            run_name = f"{algo_name}_{'_'.join([f'{k}{v}' for k,v in run_params.items()])}"
            
            with mlflow.start_run(run_name=run_name) as run:
                print(f"Training {run_name}...")
                
                model = get_model(algo_name, run_params, seed)
                model.fit(train_x, train_y)

                predicted_qualities = model.predict(test_x)
                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                print(f"  RMSE: {rmse}")
                print(f"  MAE: {mae}")
                print(f"  R2: {r2}")

                # Log params and metrics
                mlflow.log_params(run_params)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                mlflow.sklearn.log_model(model, "model")
                
                # Check if best
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_run_id = run.info.run_id
                    best_model = model
                    best_algo_name = algo_name
                    print(f"  -> New best model found!")

    if best_run_id:
        print(f"\nOptimization Complete. Best Run ID: {best_run_id} with RMSE: {best_rmse}")
        
        # Prepare Triton Model Repository structure locally
        # We start with the static repository template (configs, python backends)
        source_repo = "model_repository"
        triton_repo_path = "models/triton_repository"
        
        if os.path.exists(triton_repo_path):
            shutil.rmtree(triton_repo_path)
            
        # Copy the static repository structure
        shutil.copytree(source_repo, triton_repo_path)
        
        # Overwrite the specific model version with the new trained artifact
        model_name = "wine_model" # Must match directory name in model_repository
        version = "1"
        export_path = os.path.join(triton_repo_path, model_name, version)
        
        os.makedirs(export_path, exist_ok=True)
        
        # Convert & Save ONNX
        try:
            initial_type = [('float_input', FloatTensorType([None, 13]))]
            onx = convert_sklearn(best_model, initial_types=initial_type)
            
            onnx_path = os.path.join(export_path, "model.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())
                
            print(f"Triton-ready model saved to {export_path}")
            
            # Log the entire Triton repository as an artifact
            with mlflow.start_run(run_id=best_run_id):
                mlflow.log_artifacts(triton_repo_path, artifact_path="triton_repo")
                print("Logged Full Triton repository to MLflow")
                
            # Update artifact URI
            with mlflow.start_run(run_id=best_run_id):
                 # Get URI of the directory
                 artifact_uri = mlflow.get_artifact_uri("triton_repo")
            
            print(f"Triton Artifact URI: {artifact_uri}")
            
            import json
            with open("run_info.json", "w") as f:
                json.dump({"run_id": best_run_id, "artifact_uri": artifact_uri, "best_rmse": best_rmse}, f)

        except Exception as e:
            print(f"Warning: Failed to convert {best_algo_name} to ONNX: {e}")
            
        # Also keep the local sklearn copy for dev
        if os.path.exists("models/wine_model"):
            shutil.rmtree("models/wine_model")
        os.makedirs("models/wine_model", exist_ok=True)
        mlflow.sklearn.save_model(best_model, "models/wine_model/sklearn")
        
    else:
        print("No models were trained.")

if __name__ == "__main__":
    train_optimization()

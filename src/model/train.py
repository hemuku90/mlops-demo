import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.onnx
import sys
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model(alpha=0.5, l1_ratio=0.5):
    # Load data
    csv_url = os.path.join("data", "wine_quality.csv")
    try:
        data = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Unable to read file. Error: {e}")
        return

    # Split data
    train, test = train_test_split(data)

    train_x = train.drop(["target"], axis=1)
    test_x = test.drop(["target"], axis=1)
    train_y = train[["target"]]
    test_y = test[["target"]]

    # Set tracking URI to point to the Dockerized MLflow server by default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Logging to MLflow at {tracking_uri}")

    # Start MLflow run
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
        
        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType([None, 13]))]
        onx = convert_sklearn(lr, initial_types=initial_type)
        
        # Log ONNX model to MLflow
        mlflow.onnx.log_model(onx, "model_onnx")
        
        # Save run info for CI/CD
        run_id = mlflow.active_run().info.run_id
        artifact_uri = mlflow.get_artifact_uri("model")
        print(f"Run ID: {run_id}")
        print(f"Artifact URI: {artifact_uri}")
        
        import json
        with open("run_info.json", "w") as f:
            json.dump({"run_id": run_id, "artifact_uri": artifact_uri}, f)

        # Save model locally for easy access by app (simulating model registry fetch)
        import shutil
        if os.path.exists("models/wine_model"):
            shutil.rmtree("models/wine_model")
        os.makedirs("models/wine_model", exist_ok=True)
        mlflow.sklearn.save_model(lr, "models/wine_model/sklearn")
        
        # Save ONNX model locally
        with open("models/wine_model/model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
            
        print("Model saved to models/wine_model (sklearn and onnx)")

if __name__ == "__main__":
    train_model()

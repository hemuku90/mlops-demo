from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import uvicorn
import os
import requests
import json
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

app = FastAPI(title="Wine Quality Prediction API")

# Configuration
TRITON_URL = os.getenv("TRITON_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "models/wine_model")

model = None

if TRITON_URL:
    print(f"Configured to proxy predictions to Triton: {TRITON_URL}")
else:
    # Load model locally for Dev/Test
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print(f"Model loaded locally from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading local model: {e}")

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.get("/")
def read_root():
    mode = "Triton Proxy" if TRITON_URL else "Local Model"
    return {"message": "Wine Quality Prediction API", "mode": mode}

@app.post("/predict")
def predict(features: WineFeatures):
    data_dict = features.dict()
    
    if TRITON_URL:
        # Proxy to Triton Ensemble
        try:
            client = httpclient.InferenceServerClient(url=TRITON_URL)
            inputs = []
            
            # Create individual inputs for each feature
            for key, value in data_dict.items():
                # Input shape: [BATCH_SIZE, 1]. Here batch size is 1.
                input_data = np.array([[value]], dtype=np.float32)
                
                # Create InferInput
                infer_input = httpclient.InferInput(key, input_data.shape, "FP32")
                infer_input.set_data_from_numpy(input_data)
                inputs.append(infer_input)
            
            # Request output "prediction" from ensemble
            output = httpclient.InferRequestedOutput("prediction")
            
            # Run inference
            response = client.infer("ensemble_model", inputs=inputs, outputs=[output])
            
            # Get result
            prediction = response.as_numpy("prediction")
            
            # Result is [Batch, 1], extract scalar
            return {"prediction": float(prediction[0][0])}
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Triton inference failed: {str(e)}")

    else:
        # Local Inference (Scikit-Learn)
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded locally")
        
        # Rename column to match model training data (CSV has slash)
        if 'od280_od315_of_diluted_wines' in data_dict:
            data_dict['od280/od315_of_diluted_wines'] = data_dict.pop('od280_od315_of_diluted_wines')
        
        data = pd.DataFrame([data_dict])
        
        # Ensure columns are in the exact order the model expects
        expected_cols = [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
            "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
        ]
        data = data[expected_cols]
        
        try:
            prediction = model.predict(data)
            return {"prediction": prediction[0]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

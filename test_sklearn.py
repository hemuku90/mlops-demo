import mlflow.sklearn
import pandas as pd
import numpy as np
import os

try:
    print("Loading Sklearn model...")
    model_path = "models/wine_model/sklearn"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        exit(1)
        
    model = mlflow.sklearn.load_model(model_path)
    print("Model loaded successfully.")

    # Data from the curl request (Class 0)
    data_0 = [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
    
    # Sample 1 (Target 1) - Index 59
    data_1 = [12.37, 0.94, 1.36, 10.60, 88.00, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520.00]

    # Sample 2 (Target 2) - Index 130
    data_2 = [12.86, 1.35, 2.32, 18.00, 122.00, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29, 630.00]

    samples = [("Class 0", data_0), ("Class 1", data_1), ("Class 2", data_2)]
    
    # Feature names matching training data
    feature_names = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
    ]

    for name, data in samples:
        # Create DataFrame with feature names
        df = pd.DataFrame([data], columns=feature_names)
        
        prediction = model.predict(df)
        print(f"Prediction for {name}: {prediction[0]}")

except Exception as e:
    print(f"Error: {e}")

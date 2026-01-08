import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from alibi_detect.cd import KSDrift
from alibi_detect.utils.saving import save_detector
import os
import dill

def train_drift_detector():
    # Load data
    csv_url = os.path.join("data", "wine_quality.csv")
    try:
        data = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Unable to read file. Error: {e}")
        return

    # Split data (same random state as model training to ensure same reference data)
    train, test = train_test_split(data, random_state=None) # train.py uses default random_state which is None? No, train.py uses None in train_test_split call.

    # Features only
    X_train = train.drop(["target"], axis=1).values.astype(np.float32)
    
    # Define Drift Detector
    # K-S (Kolmogorov-Smirnov) test for feature-wise drift detection on continuous data
    # p_val: p-value used for significance of the K-S test
    cd = KSDrift(X_train, p_val=0.05)

    # Save the detector
    output_dir = "models/drift_detector"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save using dill for simple loading or alibi's save_detector
    # Seldon's alibi-detect server often expects the artifact to be loaded via dill or specific structure
    # We will save as 'detector.dill' which is a common pattern for Seldon
    filepath = os.path.join(output_dir, "detector.dill")
    with open(filepath, "wb") as f:
        dill.dump(cd, f)
    
    print(f"Drift detector saved to {filepath}")

if __name__ == "__main__":
    train_drift_detector()

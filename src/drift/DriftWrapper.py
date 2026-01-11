import dill
import logging
import os
import numpy as np

class DriftWrapper:
    def __init__(self):
        self.model_name = "DriftWrapper"
        self.detector = None
        self.ready = False
        self.load()

    def load(self):
        # STORAGE_URI is provided by Seldon
        model_path = os.getenv("STORAGE_URI", "/mnt/models/drift_detector")
        # Handle if STORAGE_URI is the folder or file
        if os.path.isdir(model_path):
            detector_path = os.path.join(model_path, "detector.dill")
        else:
            detector_path = model_path
            
        logging.info(f"Loading drift detector from: {detector_path}")
        try:
            with open(detector_path, "rb") as f:
                self.detector = dill.load(f)
            self.ready = True
            logging.info("Drift detector loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load drift detector: {e}")
            self.ready = False
            # Don't raise here, let readiness probe fail? 
            # Seldon will check health via separate endpoint, but let's be safe.
            raise e

    def predict(self, X, feature_names=None, **kwargs):
        if not self.ready:
            raise RuntimeError("Drift detector not ready")

        try:
            logging.info(f"Received request with type(X): {type(X)}")
            logging.info(f"Received request with X: {X}")
            
            # X comes from Seldon. If it's a request from another Seldon model (logger),
            # the payload structure might be complex (SeldonMessage).
            # But the Seldon Python wrapper usually unwraps 'data' -> 'ndarray'.
            
            # Ensure X is numpy array
            X = np.array(X)
            
            # Alibi Detect expects batch of samples
            # KSDrift: Input should be (N, features)
            
            preds = self.detector.predict(X)
            
            # Extract relevant metrics
            is_drift = int(preds['data']['is_drift'])
            p_vals = preds['data']['p_val']
            
            logging.info(f"Drift detection result: is_drift={is_drift}, p_vals={p_vals}")

            # If p_vals is array (feature-wise), take min or mean?
            # KSDrift returns p-values per feature.
            # We can log them as custom metrics.
            
            # Construct response
            # We return the drift status as the "prediction"
            return np.array([[is_drift]])

        except Exception as e:
            logging.error(f"Error during drift detection: {e}")
            # Return valid shape to avoid breaking caller if synchronous
            return np.array([[-1]])

    def metrics(self):
        # Custom metrics for Prometheus
        # This requires the wrapper to have processed a request recently or we need to store state?
        # Seldon calls metrics() after predict().
        # Actually, Seldon's metrics support is a bit different.
        # We can return a list of dicts.
        return [
            {"type": "GAUGE", "key": "drift_found", "value": 1} # Placeholder
        ]
        
    def tags(self):
        return {"model": "drift_detector"}

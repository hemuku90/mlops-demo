from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import NumpyCodec
import numpy as np
import dill
import os
import logging

class DriftWrapper(MLModel):
    async def load(self) -> bool:
        # STORAGE_URI is provided by Seldon/Kubernetes env or settings
        # We fallback to baked-in location
        model_path = os.getenv("STORAGE_URI", "/app/models/drift_detector")
        
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
            return True
        except Exception as e:
            logging.error(f"Failed to load drift detector: {e}")
            self.ready = False
            raise e

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        try:
            logging.info(f"Received inference request payload")
            
            # Expected feature order matching the training data and other models
            expected_cols = [
                "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
            ]
            
            # 1. Try to parse inputs as dictionary of named inputs
            inputs_map = {inp.name: inp for inp in payload.inputs}
            
            # Check if we have the expected columns or just a single input (like "data" or "payload")
            # If we have named features, we reconstruct the vector
            if any(col in inputs_map for col in expected_cols):
                logging.info("Processing named feature inputs")
                vector = []
                for col in expected_cols:
                    if col in inputs_map:
                        inp = inputs_map[col]
                        # Decode input using NumpyCodec
                        data = NumpyCodec.decode_input(inp)
                        # data is likely numpy array. Take item.
                        # Handle batch size. Assuming batch 1 for now or taking first.
                        val = data.item(0) if data.size > 0 else 0.0
                        vector.append(val)
                    else:
                        logging.warning(f"Missing feature: {col}, using 0.0")
                        vector.append(0.0)
                
                X = np.array([vector], dtype=np.float32)
            
            else:
                # Fallback: maybe single input "data" or "inputs" containing the array?
                # Or maybe the payload IS the array in one input?
                logging.info("No expected feature names found. checking first input.")
                if len(payload.inputs) > 0:
                    inp = payload.inputs[0]
                    X = NumpyCodec.decode_input(inp)
                    # Ensure shape (N, 13)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                else:
                    logging.error("Empty inputs in payload")
                    raise ValueError("Empty inputs")

            logging.info(f"Input shape for detection: {X.shape}")

            # Run Drift Detection
            # Alibi Detect predict returns a dict
            preds = self.detector.predict(X)
            
            is_drift = int(preds['data']['is_drift'])
            p_vals = preds['data']['p_val']
            
            logging.info(f"Drift detection result: is_drift={is_drift}, p_vals={p_vals}")
            
            # Construct Response
            # We return is_drift as the primary output
            return InferenceResponse(
                model_name=self.name,
                model_version=self.version,
                outputs=[
                    ResponseOutput(
                        name="is_drift",
                        datatype="INT32",
                        shape=[1],
                        data=[is_drift]
                    ),
                    ResponseOutput(
                        name="p_val",
                        datatype="FP32",
                        shape=[len(p_vals) if isinstance(p_vals, (list, np.ndarray)) else 1],
                        data=p_vals.tolist() if isinstance(p_vals, np.ndarray) else [p_vals]
                    )
                ]
            )
            
        except Exception as e:
            logging.error(f"Error during drift detection: {e}", exc_info=True)
            # Return error or empty response?
            # MLServer will handle exception raised here usually
            raise e

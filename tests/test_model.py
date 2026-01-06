import os
import pytest

def test_data_exists():
    assert os.path.exists("data/wine_quality.csv"), "Data file not found"

def test_model_artifact_exists():
    assert os.path.exists("models/wine_model"), "Model directory not found"
    assert os.path.exists("models/wine_model/model.onnx"), "ONNX model not found"
    assert os.path.exists("models/wine_model/sklearn"), "Sklearn model directory not found"

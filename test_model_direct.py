import requests
import json
import numpy as np

# Data sample (Class 1)
data = {
    "alcohol": 12.37,
    "malic_acid": 0.94,
    "ash": 1.36,
    "alcalinity_of_ash": 10.6,
    "magnesium": 88.0,
    "total_phenols": 1.98,
    "flavanoids": 0.57,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 0.42,
    "color_intensity": 1.95,
    "hue": 1.05,
    "od280_od315_of_diluted_wines": 1.82,
    "proline": 520.0
}

payload = {
    "inputs": [
        {"name": name, "shape": [1, 1], "datatype": "FP32", "data": [value]}
        for name, value in data.items()
    ]
}

url = "http://localhost:8000/v2/models/ensemble-model/infer"

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print("Response status:", response.status_code)
    print("Response body:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
    if 'response' in locals():
        print(f"Response text: {response.text}")

import onnxruntime as ort
import numpy as np
import sys

try:
    # Load the model
    print("Loading model.onnx...")
    sess = ort.InferenceSession("model.onnx")

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    # Sample 0 (Target 0) - The one the user used
    data_0 = [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
    
    # Sample 1 (Target 1) - Index 59
    data_1 = [12.37, 0.94, 1.36, 10.60, 88.00, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520.00]

    # Sample 2 (Target 2) - Index 130
    data_2 = [12.86, 1.35, 2.32, 18.00, 122.00, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29, 630.00]

    samples = [("Class 0", data_0), ("Class 1", data_1), ("Class 2", data_2)]

    for name, data in samples:
        input_data = np.array([data], dtype=np.float32)
        result = sess.run([output_name], {input_name: input_data})
        print(f"Prediction for {name}: {result[0][0]}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

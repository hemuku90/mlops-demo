import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = bs_utils = args['model_config']

    def execute(self, requests):
        responses = []
        for request in requests:
            # List of input feature names in specific order
            feature_names = [
                "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
            ]
            
            # Extract inputs
            input_data = []
            batch_size = 0
            
            # We assume all inputs have the same batch size
            first_input = pb_utils.get_input_tensor_by_name(request, feature_names[0])
            if first_input is not None:
                batch_size = first_input.as_numpy().shape[0]
            
            # Stack features: [Batch, 1] -> [Batch, 13]
            # Initialize empty array [Batch, 13]
            processed_data = np.zeros((batch_size, len(feature_names)), dtype=np.float32)
            
            for i, name in enumerate(feature_names):
                tensor = pb_utils.get_input_tensor_by_name(request, name)
                if tensor is not None:
                    # tensor.as_numpy() is [Batch, 1]
                    processed_data[:, i] = tensor.as_numpy().flatten()
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("float_input", processed_data)
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
            
        return responses

    def finalize(self):
        pass

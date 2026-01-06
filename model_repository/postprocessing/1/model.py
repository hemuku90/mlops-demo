import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = bs_utils = args['model_config']

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get model output
            # "variable" is the output name from the ONNX model
            input_tensor = pb_utils.get_input_tensor_by_name(request, "variable")
            
            if input_tensor is not None:
                # Just pass it through as "prediction"
                # In a more complex scenario, we could apply thresholds, map to classes, etc.
                prediction_data = input_tensor.as_numpy()
                
                output_tensor = pb_utils.Tensor("prediction", prediction_data)
                
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)
            else:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Input tensor 'variable' not found")
                ))
            
        return responses

    def finalize(self):
        pass

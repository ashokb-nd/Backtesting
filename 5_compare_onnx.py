#compares to onnx files. with random inputs

import onnxruntime as ort
import numpy as np

def load_model(model_path):
    return ort.InferenceSession(model_path)

def generate_random_input(input_shape):
    return np.random.rand(*input_shape).astype(np.float32)

def get_model_output(model, input_data):
    input_name = model.get_inputs()[0].name
    return model.run(None, {input_name: input_data})


# Load the models
model1_path = '/data4/ashok/dr_distraction/ImprovingDriverDistraction_AN_19312/data/0_production_model_data/distracted_multihead_v0.5.3/onnxmodel'
model2_path = '/data4/ashok/dr_distraction/ImprovingDriverDistraction_AN_19312/data/v5_final/after_retraining/FINAL_bn_copied.onnx'
model1 = load_model(model1_path)
model2 = load_model(model2_path)

# Generate random input
input_shape = model1.get_inputs()[0].shape
random_input = generate_random_input(input_shape)

# Get outputs from both models
output1 = get_model_output(model1, random_input)
output2 = get_model_output(model2, random_input)

# Compare the outputs
for i in range(2):
    print(np.allclose(output1[i], output2[i], atol=1e-10))
    print(f"diff: {np.abs(output1[i] - output2[i]).max()}")
import torch
from tensorflow.python import Variable

from core.main import VideoGen
import pytorch2keras
import numpy as np

def save():
    PATH = './pytorch_model/test_gen.pt'

    model = VideoGen().float()
    model.load_state_dict(torch.load(PATH))

    input_np = np.random.uniform(0, 1, (1, 3, 64, 64))
    input_var = torch.FloatTensor(input_np)

    torch.onnx.export(model, input_var, "testing.onnx", verbose=True)

import onnx

# Load the ONNX model
model = onnx.load("./onxx_models/testing.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
import onnx
from onnx import numpy_helper
import numpy as np
# from tvm import relay

# Define the operation (addition in this case)
# node = onnx.helper.make_node(
#     'Add',
#     inputs=['input1', 'input2'],
#     outputs=['output'],
# )

# # Define the input tensors
# input1 = onnx.helper.make_tensor_value_info('input1', onnx.TensorProto.FLOAT, [1,3,4,4])
# input2 = onnx.helper.make_tensor_value_info('input2', onnx.TensorProto.FLOAT, [1,3,4,4])

# # Define the output tensor
# output = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1,3,4,4])

# # Define the graph with one node
# graph = onnx.helper.make_graph([node], 'simple_addition', [input1, input2], [output])

# # Define the model with the graph
# model = onnx.helper.make_model(graph)

# # Save the model to a file
# onnx.save(model, 'simple_addition.onnx')



# node = onnx.helper.make_node(
#     'Reshape',
#     inputs=['input1'],
#     outputs=['output'],
# )

# # Define the input tensors
# input1 = onnx.helper.make_tensor_value_info('input1', onnx.TensorProto.FLOAT, [1,3,4,4])

# # Define the output tensor
# output = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1,4,4,3])
# node.attribute.append(onnx.helper.make_attribute('newshape',[1,4,4,3]))

# # Define the graph with one node
# graph = onnx.helper.make_graph([node], 'justreshape', [input1], [output])

# # Define the model with the graph
# model = onnx.helper.make_model(graph)

# # Save the model to a file
# onnx.save(model, 'justreshape.onnx')






import onnx
from onnx import numpy_helper

# Define the input tensor shape
input_shape = [3,250,2,500]

# Define the target shape for reshaping
target_shape = [1,3,500,500]

# Create a Constant node for the target shape
target_shape_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['target_shape'],
    value=numpy_helper.from_array(np.array(target_shape, dtype=np.int64)),
)

# Define the Reshape node
reshape_node = onnx.helper.make_node(
    'Reshape',
    inputs=['input', 'target_shape'],
    outputs=['output'],
)

# Define the input tensor
input_tensor = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input_shape)

# Define the output tensor
output_tensor = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, target_shape)

# Define the graph with the nodes
graph = onnx.helper.make_graph([target_shape_node, reshape_node], 'simple_reshape', [input_tensor], [output_tensor])

# Define the model with the graph
model = onnx.helper.make_model(graph)

# Save the model to a file
onnx.save(model, 'justreshape.onnx')

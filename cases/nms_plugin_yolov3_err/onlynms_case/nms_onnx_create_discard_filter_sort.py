# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import onnx
import numpy as np
from onnx import helper, numpy_helper
import pdb



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Create input data
    # input1_shape = (3200,4)
    boxes_shape = (1,1,3200,4) # box不够
    boxes_data = np.arange(np.prod(boxes_shape)).reshape(boxes_shape)
    boxes_tensor = helper.make_tensor_value_info('origin_boxes', onnx.TensorProto.FLOAT, boxes_data.shape)

    sortedscore_shape = (1,8,4096) # sorted scores 没给
    sortedscore_data = np.arange(np.prod(sortedscore_shape)).reshape(sortedscore_shape)
    sortedscore_tensor = helper.make_tensor_value_info('sorted_scores', onnx.TensorProto.FLOAT, sortedscore_data.shape)

    # pdb.set_trace()
    # Create constant data
    # input2_data_shape = (8,4096)
    sortedidx_shape = (1,8,4096)
    sortedidx_data = np.arange(np.prod(sortedidx_shape)).reshape(sortedidx_shape)
    sortedidx_tensor = helper.make_tensor_value_info('sorted_idx', onnx.TensorProto.INT32, sortedidx_data.shape)

    sortsize_data_shape = (1,8)
    sortsize_data = np.arange(np.prod(sortsize_data_shape)).reshape(sortsize_data_shape)
    sortsize_tensor = helper.make_tensor_value_info('sort_size', onnx.TensorProto.INT32, sortsize_data.shape)

    # Create output data
    # output_data_shape = (100,200,300)
    # output_data = np.arange(np.prod(output_data_shape)).reshape(output_data_shape)
    # output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output_data.shape)

    class_idx   = helper.make_tensor_value_info('classes_idx', onnx.TensorProto.INT32, [1, 200, 2])
    nmsed_box     = helper.make_tensor_value_info('nmsed_boxes', onnx.TensorProto.FLOAT, [1, 200, 4])
    nmsed_score   = helper.make_tensor_value_info('nmsed_scores', onnx.TensorProto.FLOAT, [1, 200, 1])
    num_detection = helper.make_tensor_value_info('num_detections', onnx.TensorProto.INT32, [1, 1])

    # Create Add operator node
    nms_node = helper.make_node(
        'DLNonMaxSuppression',
        inputs=['origin_boxes','sorted_scores', 'sorted_idx','sort_size'],
        # outputs=['output'],
        outputs=['classes_idx', 'nmsed_boxes', 'nmsed_scores', 'num_detections']
    )

    nms_node.attribute.append(helper.make_attribute('backgroundLabelId', -1))
    nms_node.attribute.append(helper.make_attribute('iouThreshold', 0.45))
    nms_node.attribute.append(helper.make_attribute('isNormalized', False))
    # nms_node.attribute.append(helper.make_attribute('keepTopK', 200))
    nms_node.attribute.append(helper.make_attribute('numClasses', 8)) #
    nms_node.attribute.append(helper.make_attribute('shareLocation', True))
    nms_node.attribute.append(helper.make_attribute('scoreThreshold', 0.25))
    # nms_node.attribute.append(helper.make_attribute('topK', 1000))

    # Create graph
    graph_def = helper.make_graph(
        nodes=[nms_node],
        name='single_operator_network',
        inputs=[boxes_tensor, sortedscore_tensor, sortedidx_tensor, sortsize_tensor],
        # outputs=[output_tensor],
        outputs=[class_idx, nmsed_box, nmsed_score, num_detection]
    )

    # Create model
    model_def = helper.make_model(graph_def, producer_name='SingleOperatorNetwork')

    # Save model to ONNX file
    onnx.save(model_def, 'onlynmsplugin_discard_filtersort.onnx')
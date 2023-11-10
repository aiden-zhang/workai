import numpy as np
import onnxruntime as ort
import onnx

from onnx import GraphProto, TensorProto
from onnx.helper import make_tensor_value_info, make_attribute, make_model, make_node

dynamic_batch = False

def append_nms(graph, unused_node=[]):
    ngraph = GraphProto()
    ngraph.name = graph.name

    ngraph.input.extend([i for i in graph.input if i.name not in unused_node])
    ngraph.initializer.extend([i for i in graph.initializer if i.name not in unused_node])
    ngraph.value_info.extend([i for i in graph.value_info if i.name not in unused_node])
    ngraph.node.extend([i for i in graph.node if i.name not in unused_node])

    output_info = [i for i in graph.output]
    ngraph.value_info.extend(output_info)
    
    print(ngraph.output)

    nms = make_node(
        'DLNonMaxSuppression',
        inputs = ['bboxes', 'classes'],
        outputs = ['num_detections', 'nmsed_boxes', 'nmsed_scores', 'nmsed_classes']
    )
    nms.attribute.append(make_attribute('backgroundLabelId', -1))
    nms.attribute.append(make_attribute('iouThreshold', 0.45))
    nms.attribute.append(make_attribute('isNormalized', False))
    nms.attribute.append(make_attribute('keepTopK', 200))
    nms.attribute.append(make_attribute('numClasses', 80)) #
    nms.attribute.append(make_attribute('shareLocation', True))
    nms.attribute.append(make_attribute('scoreThreshold', 0.25))
    nms.attribute.append(make_attribute('topK', 1000))
    ngraph.node.append(nms)

    if dynamic_batch:
        num_detection = make_tensor_value_info('num_detections', TensorProto.INT32, ['-1', 1])
        nmsed_box = make_tensor_value_info('nmsed_boxes', TensorProto.FLOAT, ['-1', 200, 4])
        nmsed_score = make_tensor_value_info('nmsed_scores', TensorProto.FLOAT, ['-1', 200, 1])
        nmsed_class = make_tensor_value_info('nmsed_classes', TensorProto.FLOAT, ['-1', 200, 1])
    else:
        num_detection = make_tensor_value_info('num_detections', TensorProto.INT32, [1, 1])
        nmsed_box = make_tensor_value_info('nmsed_boxes', TensorProto.FLOAT, [1, 200, 4])
        nmsed_score = make_tensor_value_info('nmsed_scores', TensorProto.FLOAT, [1, 200, 1])
        nmsed_class = make_tensor_value_info('nmsed_classes', TensorProto.FLOAT, [1, 200, 1])

    ngraph.output.extend([num_detection, nmsed_box, nmsed_score, nmsed_class])

    return ngraph

if __name__ == '__main__':
    model = onnx.load('./yolov5s.onnx')

    model_attrs = dict(
        ir_version = model.ir_version,
        opset_imports = model.opset_import,
        producer_version = model.producer_version,
        model_version = model.model_version
    )

    model = make_model(append_nms(model.graph), **model_attrs)

    onnx.save(model, 'yolov5s_nms.onnx')


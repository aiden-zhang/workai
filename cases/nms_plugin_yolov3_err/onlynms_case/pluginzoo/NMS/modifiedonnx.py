import numpy as np
import onnxruntime as ort
import onnx
from onnx import defs, checker, helper, numpy_helper, mapping
from onnx import ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorProto, OperatorSetIdProto
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node

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
    
    # box_starts = make_tensor('box_starts', onnx.TensorProto.INT64, [1], [0])
    # box_ends = make_tensor('box_ends', onnx.TensorProto.INT64, [1], [4])
    # box_axes = make_tensor('box_axes', onnx.TensorProto.INT64, [1], [2])
    # box_steps = make_tensor('box_steps', onnx.TensorProto.INT64, [1], [1])
    # ngraph.initializer.append(box_starts)
    # ngraph.initializer.append(box_ends)
    # ngraph.initializer.append(box_axes)
    # ngraph.initializer.append(box_steps)
    # node_boxes = make_node('Slice', inputs=[b'output', 'box_starts', 'box_ends', 'box_axes', 'box_steps'], 
    #                         outputs=['boxes'], name='slice_box')
    # ngraph.node.append(node_boxes)
    
    # score_starts = make_tensor('score_starts', onnx.TensorProto.INT64, [1], [4])
    # score_ends = make_tensor('score_ends', onnx.TensorProto.INT64, [1], [5])
    # score_axes = make_tensor('score_axes', onnx.TensorProto.INT64, [1], [2])
    # score_steps = make_tensor('score_steps', onnx.TensorProto.INT64, [1], [1])
    # ngraph.initializer.append(score_starts)
    # ngraph.initializer.append(score_ends)
    # ngraph.initializer.append(score_axes)
    # ngraph.initializer.append(score_steps)
    # node_score = make_node('Slice', 
    #             inputs=['output', 'score_starts', 'score_ends', 'score_axes', 'score_steps'],
    #             outputs=['scores'])
    # ngraph.node.append(node_score)
   
   
    # class_starts = make_tensor('class_starts', onnx.TensorProto.INT64, [1], [5])
    # class_ends = make_tensor('class_ends', onnx.TensorProto.INT64, [1], [85])
    # class_axes = make_tensor('class_axes', onnx.TensorProto.INT64, [1], [2])
    # class_steps = make_tensor('class_steps', onnx.TensorProto.INT64, [1], [1])
    # ngraph.initializer.append(class_starts)
    # ngraph.initializer.append(class_ends)
    # ngraph.initializer.append(class_axes)
    # ngraph.initializer.append(class_steps)
    # node_classe = make_node('Slice', 
    #             inputs=['output', 'class_starts', 'class_ends', 'class_axes', 'class_steps'],
    #             outputs=['classes'], name='slice_class')
    # ngraph.node.append(node_classe)
    
    # uns_axes = make_tensor('uns_axes', onnx.TensorProto.INT64, [1], [0])
    # ngraph.initializer.append(uns_axes)
    # uns_node = onnx.helper.make_node(
    #     "Unsqueeze",
    #     inputs=["boxes", "uns_axes"],
    #     outputs=["bboxes"],
    # )
    # ngraph.node.append(uns_node)
    
    # ( 1,25200,4) ->(1, 1,25200,4)
    # boxes = make_tensor_value_info('boxes', onnx.TensorProto.FLOAT, shape = (1, 1,25200,4))
    # scores = make_tensor_value_info('scores', onnx.TensorProto.FLOAT, shape = (1,25200,1))
    # classes = make_tensor_value_info('classes', onnx.TensorProto.FLOAT, shape = (1,25200,80))
    
    
    # ngraph.output.extend([boxes, scores, classes])
    # ngraph.output.extend([boxes, classes])
    
    # return ngraph

    # # ngraph.value_info.append(make_tensor_value_info(score_node, TensorProto.FLOAT, [1, 2134, 3]))
    # # ngraph.value_info.append(make_tensor_value_info(bbox_node, TensorProto.FLOAT, [1, 2134, 1, 4]))

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
    # print(model.graph)
    # checker.check_model(model)
    
    
    onnx.save(model, 'yolov5s_nms.onnx')
    
    sess = ort.InferenceSession('yolov5s_nms.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    images = np.random.randint(0,1,(1,3,640,640)).astype(np.float32)
    outputs = sess.run(['boxes', 'classes'], {input_name:images})
    
    # print(outputs)


from nne_network_utils import *
import pytest
from tensorflow.core.protobuf import config_pb2
from pathlib import Path
import dlnne as nne
from golden_runner import run_onnx
from golden_runner import run_tf
from golden_runner import run_tflite
import time

tf_gpu_options = config_pb2.GPUOptions(allow_growth=True)
tf_config = config_pb2.ConfigProto(gpu_options=tf_gpu_options, allow_soft_placement=True)

quantized_pb_dir = "/mars/share/DLI/models/quantized/"
frozen_pb_dir = "/mars/share/DLI/models/frozen/"
info_file = "/mars/share/DLI/models/network_interface.txt"

models_list = [
    'alexnet_frozen_graph.pb',
    'bert_frozen_graph.pb',
    'bert_frozen_graph_fp16.pb',
    'bert_frozen_graph_fp16_fold_constants.pb',
    'bert_frozen_graph_fp16_bias_fp32_fold_constants.pb',
    'C3D_sports1m_finetuning_ucf101.pb',
    'cosface_frozen_graph.pb',  # Mismatch
    'CRNN_frozen_graph.pb',
    'ctpn_frozen_graph.pb',
    'cws_wemb_frozen_graph.pb',
    'deeplab_v3_frozen_graph.pb',
    'deepspeech_v1_frozen_graph.pb',
    'densenet169_frozen_graph.pb',
    'dssm_frozen_graph.pb',
    'facenet_frozen_graph.pb',
    'faster_rcnn_vgg_frozen_graph.pb',
    'fpn_resnet_v1_frozen_graph.pb',
    'gnmt_frozen_graph.pb',
    'gpt_frozen_graph.pb',
    'gpt_tu_fp16_bias_fp16_fold_constants.pb',
    'inception_v1_frozen_graph.pb',
    'inception_v2_frozen_graph.pb',
    'inception_v3_frozen_graph.pb',
    'inception_v3_pruning_frozen_graph.pb',
    'inception_v3_tu_fp16_bias_fp32_fold_constants.pb',
    'MaskRCNN_frozen_graph.pb',
    'meta_bilstm_frozen_graph.pb',
    'NBT_CNN_frozen_graph.pb',
    'inception_v4_frozen_graph.pb',
    'mobilenet_v1_frozen_graph.pb',
    'mobilenet_v2_frozen_graph.pb',
    'nasnet_large_frozen_graph.pb',
    'nasnet_mobile_frozen_graph.pb',
    'NCF_frozen_graph.pb',
    'nin_frozen_graph_without_training.pb',
    'ParseLM_frozen_graph.pb',
    'PSPNet_frozen_graph.pb',
    'question_answering_frozen_graph.pb',
    'R2CNN_resnet_v1_frozen_graph.pb',
    'ranknet_frozen_graph.pb',
    'resnet_v1_101_frozen_graph.pb',
    'resnet_v1_152_frozen_graph.pb',
    'resnet_v1_50_frozen_graph.pb',
    'resnet_v2_50_frozen_graph.pb',
    'wide_and_deep_origin_tf23.pb',

    'resnext101_32x4d_bottleneck_type_b_frozen_graph_without_training.pb',
    'resnext101_32x4d_bottleneck_type_c_frozen_graph_without_training.pb',
    'resnext50_32x4d_bottleneck_type_b_frozen_graph_without_training.pb',
    'resnext50_32x4d_bottleneck_type_c_frozen_graph_without_training.pb',
    'resnext101_32x4d_type_c_group_conv_frozen_graph_without_training.pb',
    'SegNet_frozen_graph_without_training.pb',
    'simple_network_frozen_graph.pb',
    'summarization_frozen_graph.pb',
    'transformer_frozen_graph.pb',
    'shufflenet_v1_frozen_graph_without_training.pb',
    'shufflenet_v2_frozen_graph.pb',
    'squeezenet_v1_frozen_graph.pb',
    'ssd_vgg_300_frozen_graph.pb',
    'vgg16_frozen_graph.pb',
    'vgg_16_pruning_frozen_graph.pb',
    'xception_frozen_graph.pb',
    'stn_frozen_graph.pb',
    'deepspeech_v1_dynamic_quant_tu.pb',
    'HyTE_frozen_graph.pb',
    'mobilenet_v1_tu_fp16_bias_fp16_fold_constants.pb',
    'cosface_origin_tu_fp16_fold_constants.pb',
    'cosface_origin_fake_quantized_graph_tu.pb',
    'cosface_fake_quantized_graph.pb',
    'resnext50_32x4d_type_c_fake_quantized_graph.pb',
    'yolo_v3_frozen_graph.pb',
    'Mixnet_L_frozen_graph_no_swish_f32.pb',
    'yolov4_frozen_graph.pb',
    'mobilenext_without_training_frozen_graph.pb',
    'efficientdet-d0_frozen.pb',
    'efficientdet-d4_frozen.pb',
    'efficientdet-d6_frozen.pb',
    # null test
    'gnmt',
    # tflite test
    "resnet_v1_50_model.tflite",
    "vgg16_model.tflite",
    "mobilenet_v1_model.tflite",

    # onnx test
    "densenet201_224_224_without_trainning.onnx",
    "densenet-9.onnx",
    "inception-v1-9.onnx",
    "inceptionv3_224_224.onnx",
    "mobilenetv2-7.onnx",
    "mobilenet_224_224.onnx",
    "resnet18-v1-7.onnx",
    "resnet50-caffe2-v1-9.onnx",
    "segnet_without_Aten.onnx",
    "senet-50_224_224.onnx",
    "squeezenet1.0-9.onnx",
    "squeezenet1.1-7.onnx",
    "vgg16-bn-7.onnx",
    "yolov2-coco-9.onnx",
    "yolov3-10.onnx",
    "yolov3_mobilev2_416_416.onnx",
    "yolov3_tiny_1088_1920.onnx",
    "yolov3_tiny_416_416.onnx",
    "yolov3_tiny_544_960.onnx",
    "yolov3_tiny_608_608.onnx",
    "yolov3_tiny_736_1280.onnx",
    "lstm_block_cell.pb",
    "segnet_224_224_without_trainning.onnx",
    "bvlcalexnet-9.onnx",
    "shufflenet-v2-10.onnx",
    "FasterRCNN-10.onnx",
    "ssd-10.onnx",
    "yolov3_mobile_416_416.onnx",
    "yolov3_mobile_540_960.onnx",
    "yolov3_mobile_608_608.onnx",
    "yolov3_mobile_736_1280.onnx",
    "yolov3_mobilev2_540_960.onnx",
    "yolov3_mobilev2_608_608.onnx",
    "yolov3_mobilev2_736_1280.onnx",
    "resnet_v1_50_quant_tu.pb",
    "deepspeech_v2_frozen_graph_without_training.pb",
    "quantized_mobilenet_v2_quant_tu.pb",
    "quantized_mobilenet_v1_perchannel.pb",
    "non_local_resnet18_frozen_graph_without_training.pb",
    "unet_frozen_graph.pb",
    "bert_large_frozen_graph.pb",
    "transformer-xl_frozen_graph.pb",
    "SEResNeXt_frozen_graph.pb",
    "vgg19_frozen_graph.pb",
    "yolov3_tiny_frozen_graph.pb",
    "ssd_resnet50_frozen_graph.pb",
    "ssd_resnet34_frozen_graph.pb",
    "efficientnet_B0_frozen_graph_no_swish_f32.pb",
    "efficientnet_B4_frozen_graph_no_swish_f32.pb",
    'yolo_v1_frozen_graph.pb',
    'resnet50_v1.pb',
    'yolov5s.onnx',
    'gpt2_fp32.onnx',
    'gpt2_fp16.onnx',
    'PointPillars_opt.pb',
    'dlrm_post.onnx',
    'vilbert-multi-task.onnx',
    'cycleGAN_net_G.onnx',
    'hifi_gan.onnx',
    'shanmaDetBatch1_v10_1.8.0_test.onnx',
    'shanmaDetBatch1_v9_1.8.0_test.onnx',
    'hifinet.onnx',
    'SFDDetector.onnx',
    'Wav2Lip.onnx',
    'edvr_part1.onnx',
    'edvr_part1_real_w_n_3.onnx',
    'part4_wo_stack.onnx',
    'edvr_part4_after_pixel_shuffle_wo_add.onnx',
    'pcd_cascading_n3.onnx',
    'pcd_fea_fusion_n_3.onnx',
    'offset_and_mask_cas.onnx',
    'offset_and_mask_l1.onnx',
    'offset_and_mask_l2.onnx',
    'offset_and_mask_l3.onnx',
    'pcd_l1_n_3.onnx',
    'pcd_l2_n_3.onnx',
    'clip_vit_new_fp16.onnx',
    'pcd_l3_n_3.onnx',
    'slowfast_8x8_r50.onnx'
]

def get_out_node(pb):
    """ Information for models in ai/models.git/frozen """
    output_op_name = []
    if pb == 'alexnet_frozen_graph.pb':
        output_op_name = ['alexnet/fc8/xw_plus_b']
    elif pb == 'C3D_sports1m_finetuning_ucf101.pb':
        output_op_name = ['Softmax']
    elif pb == 'cosface_frozen_graph.pb':
        output_op_name = ['conv4_/conv4_23/Add_2']
    elif pb == 'CRNN_frozen_graph.pb':
        output_op_name = ['SparseToDense']
    elif pb == 'ctpn_frozen_graph.pb':
        output_op_name = ['Reshape']
    elif pb == 'cws_wemb_frozen_graph.pb':
        output_op_name = ['fully_connected/Identity']
    elif pb == 'deeplab_v3_frozen_graph.pb':
        output_op_name = ['SemanticPredictions']
    elif pb == 'deepspeech_v1_frozen_graph.pb':
        assert 0
    elif pb == 'deepspeech_v2_frozen_graph_without_training.pb':
        output_op_name = ['dense/BiasAdd']
    elif pb == 'densenet169_frozen_graph.pb':
        output_op_name = ['densenet169/predictions/Reshape_1']
    elif pb == 'dssm_frozen_graph.pb':
        output_op_name = ['dense_layer_2/dense_layer_2_output']
    elif pb == 'facenet_frozen_graph.pb':
        output_op_name = ['SemanticPredictions']
    elif pb == 'faster_rcnn_vgg_frozen_graph.pb':
        output_op_name = ['cls_prob']
    elif pb == 'fpn_resnet_v1_frozen_graph.pb':
        output_op_name = ['fast_rcnn_predict/fast_rcnn_proposals/strided_slice_1']
    elif pb == 'gnmt_frozen_graph.pb':
        output_op_name = ['index_to_string_Lookup']
    elif pb == 'gpt_frozen_graph.pb' or pb == 'gpt_tu_fp16_bias_fp16_fold_constants.pb':
        output_op_name = ['model/Reshape_5']
    elif pb == 'inception_v3_frozen_graph.pb':
        output_op_name = ['InceptionV3/Predictions/Reshape_1']
    elif pb == 'MaskRCNN_frozen_graph.pb':
        assert 0
    elif pb == 'meta_bilstm_frozen_graph.pb':
        assert 0
    elif pb == 'mobilenet_v1_frozen_graph.pb':
        output_op_name = ['MobilenetV1/Predictions/Reshape_1']
    elif pb == 'mobilenet_v2_frozen_graph.pb':
        output_op_name = ['MobilenetV2/Predictions/Reshape_1']
    elif pb == 'NBT_CNN_frozen_graph.pb':
        output_op_name = ['food/belief_update/slot_distribution']
    elif pb == 'nin_frozen_graph_without_training.pb':
        output_op_name = ['nin/conv3/global_average']
    elif pb == 'PSPNet_frozen_graph.pb':
        output_op_name = ['conv6/BiasAdd']
    elif pb == 'R2CNN_resnet_v1_frozen_graph.pb':
        assert 0
    elif pb == 'ranknet_frozen_graph.pb':
        output_op_name = ['ranking/score']
    elif pb == 'resnet_v1_50_frozen_graph.pb':
        output_op_name = ['resnet_v1_50/SpatialSqueeze']
    elif pb == 'resnet_v2_50_frozen_graph.pb':
        output_op_name = ['resnet_v2_50/SpatialSqueeze']
    elif pb == 'resnext101_32x4d_bottleneck_type_b_frozen_graph_without_training.pb':
        output_op_name = ['resnext101_32x4d/logits/Add']
    elif pb == 'resnext101_32x4d_bottleneck_type_c_frozen_graph_without_training.pb':
        output_op_name = ['resnext101_32x4d/logits/Add']
    elif pb == 'resnext50_32x4d_bottleneck_type_b_frozen_graph_without_training.pb':
        output_op_name = ['resnext50_32x4d/logits/Add']
    elif pb == 'resnext50_32x4d_bottleneck_type_c_frozen_graph_without_training.pb':
        output_op_name = ['resnext50_32x4d/logits/Add']
    elif pb == 'shufflenet_v1_frozen_graph_without_training.pb':
        output_op_name = ['Stage5/fully_connected/BiasAdd']
    elif pb == 'shufflenet_v2_frozen_graph.pb':
        output_op_name = ['classifier/BiasAdd']
    elif pb == 'simple_network_frozen_graph.pb':
        assert 0
    elif pb == 'squeezenet_v1_frozen_graph.pb':
        output_op_name = ['squeezenet_v1_1/Squeeze']
    elif pb == 'ssd_vgg_300_frozen_graph.pb':
        output_op_name = ['predictions']
    elif pb == 'stn_frozen_graph.pb':
        output_op_name = ['stn_cnn/CNN/fully_connected_2/BiasAdd']
    elif pb == 'transformer_frozen_graph.pb':
        output_op_name = ['ToInt32']
    elif pb == 'vgg16_frozen_graph.pb':
        output_op_name = ['vgg_16/fc8/squeezed']
    elif pb == 'xception_frozen_graph.pb':
        output_op_name = ['dense/Softmax']
    elif pb == 'yolo_v1_frozen_graph.pb':
        output_op_name = ['output']
    elif pb == 'yolo_v2_frozen_graph.pb':
        output_op_name = ['output']
    elif pb == 'yolo_v3_frozen_graph.pb':
        output_op_name = ['NMS/output_boxes']
    elif pb == 'inception_v1_frozen_graph.pb':
        output_op_name = ['InceptionV1/Logits/Predictions/Reshape_1']
    elif pb == 'inception_v2_frozen_graph.pb':
        output_op_name = ['InceptionV2/Predictions/Reshape_1']
    elif pb == 'inception_v3_pruning_frozen_graph.pb':
        output_op_name = ['InceptionV3/Predictions/Reshape_1']
    elif pb == 'inception_v3_tu_fp16_bias_fp32_fold_constants.pb':
        output_op_name = ['InceptionV3/Logits/SpatialSqueeze']
    elif pb == 'bert_frozen_graph.pb' or pb == 'bert_frozen_graph_fp16.pb' or pb == 'bert_frozen_graph_fp16_fold_constants.pb' or pb == 'bert_frozen_graph_fp16_bias_fp32_fold_constants.pb':
        output_op_name = ['loss/Softmax']
    elif pb == 'inception_v4_frozen_graph.pb':
        output_op_name = ['InceptionV4/Logits/Predictions']
    elif pb == 'resnext101_32x4d_type_c_group_conv_frozen_graph_without_training.pb':
        output_op_name = ['resnext101_32x4d_type_c/logits/Add']
    elif pb == 'vgg_16_pruning_frozen_graph.pb':
        output_op_name = ['vgg_16/fc8/squeezed']
    elif pb == 'resnet_v1_101_frozen_graph.pb':
        output_op_name = ['resnet_v1_101/SpatialSqueeze']
    elif pb == 'resnet_v1_152_frozen_graph.pb':
        output_op_name = ['resnet_v1_152/SpatialSqueeze']
    elif pb == 'HyTE_frozen_graph.pb':
        output_op_name = ['Sum_3']
    elif pb == 'nasnet_large_frozen_graph.pb':
        output_op_name = ['final_layer/predictions']
    elif pb == 'nasnet_mobile_frozen_graph.pb':
        output_op_name = ['final_layer/predictions']
    elif pb == 'NCF_frozen_graph.pb':
        output_op_name = ['Sigmoid']
    elif pb == 'deepspeech_v1_dynamic_quant_tu.pb':
        output_op_name = ['act_quant/FakeQuantWithMinMaxVars_cast']
    elif pb == 'mobilenet_v1_tu_fp16_bias_fp16_fold_constants.pb':
        output_op_name = ['MobilenetV1/Logits/SpatialSqueeze']
    elif pb == 'cosface_origin_tu_fp16_fold_constants.pb':
        output_op_name = ['feature/dense/BiasAdd']
    elif pb == 'resnet_v1_50_model.tflite':
        output_op_name = ["resnet_v1_50/SpatialSqueeze"]
    elif pb == 'vgg16_model.tflite':
        output_op_name = ["vgg_16/fc8/squeezed"]
    elif pb == 'mobilenet_v1_model.tflite':
        output_op_name = ["MobilenetV1/Predictions/Reshape_1"]
    elif pb == 'densenet201_224_224_without_trainning.onnx':
        output_op_name = []
    elif pb == 'densenet-9.onnx':
        output_op_name = []
    elif pb == 'inception-v1-9.onnx':
        output_op_name = []
    elif pb == 'inceptionv3_224_224.onnx':
        output_op_name = []
    elif pb == 'mobilenetv2-7.onnx':
        output_op_name = []
    elif pb == 'mobilenet_224_224.onnx':
        output_op_name = []
    elif pb == 'resnet18-v1-7.onnx':
        output_op_name = []
    elif pb == 'resnet50-caffe2-v1-9.onnx':
        output_op_name = []
    elif pb == 'segnet_without_Aten.onnx':
        output_op_name = []
    elif pb == 'senet-50_224_224.onnx':
        output_op_name = []
    elif pb == 'squeezenet1.0-9.onnx':
        output_op_name = []
    elif pb == 'squeezenet1.1-7.onnx':
        output_op_name = []
    elif pb == 'vgg16-bn-7.onnx':
        output_op_name = []
    elif pb == 'yolov2-coco-9.onnx':
        output_op_name = []
    elif pb == 'yolov3-10.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobilev2_416_416.onnx':
        output_op_name = []
    elif pb == 'yolov3_tiny_1088_1920.onnx':
        output_op_name = []
    elif pb == 'yolov3_tiny_416_416.onnx':
        output_op_name = []
    elif pb == 'yolov3_tiny_544_960.onnx':
        output_op_name = []
    elif pb == 'yolov3_tiny_608_608.onnx':
        output_op_name = []
    elif pb == 'yolov3_tiny_736_1280.onnx':
        output_op_name = []
    elif pb == 'lstm_block_cell.pb':
        output_op_name = []
    elif pb == '"resnet_v1_50_quant_tu.pb"':
        output_op_name = []
    elif pb == 'deepspeech_v2_frozen_graph_without_training.pb':
        output_op_name = []
    elif pb == 'bert_frozen_graph_fp16_bias_fp32_fold_constants.pb':
        output_op_name = []
    elif pb == 'quantized_mobilenet_v2_quant_tu.pb':
        output_op_name = []
    elif pb == 'quantized_mobilenet_v1_perchannel.pb':
        output_op_name = []
    elif pb == 'Mixnet_L_frozen_graph_no_swish_f32.pb':
        output_op_name = []
    elif pb == 'yolov4_frozen_graph.pb':
        output_op_name = []
    elif pb == 'mobilenext_without_training_frozen_graph.pb':
        output_op_name = []
    elif pb == 'efficientdet-d0_frozen.pb':
        output_op_name = []
    elif pb == 'efficientdet-d4_frozen.pb':
        output_op_name = []
    elif pb == 'efficientdet-d6_frozen.pb':
        output_op_name = []
    elif pb == 'segnet_224_224_without_trainning.onnx':
        output_op_name = []
    elif pb == 'bvlcalexnet-9.onnx':
        output_op_name = []
    elif pb == 'shufflenet-v2-10.onnx':
        output_op_name = []
    elif pb == 'FasterRCNN-10.onnx':
        output_op_name = []
    elif pb == 'ssd-10.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobile_416_416.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobile_540_960.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobile_608_608.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobile_736_1280.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobilev2_540_960.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobilev2_608_608.onnx':
        output_op_name = []
    elif pb == 'yolov3_mobilev2_736_1280.onnx':
        output_op_name = []
    elif pb == 'resnet50_v1.pb':
        output_op_name = ['softmax_tensor']
    elif pb == 'gpt2_fp32.onnx':
        output_op_name = ['2940','2941']
    elif pb == 'gpt2_fp16.onnx':
        output_op_name = ['2940','2941']
    elif pb == 'hifi_gan.onnx':
        output_op_name = ['output']
    elif pb == 'hifinet.onnx':
        output_op_name = ['output']
    elif pb == 'PointPillars_opt.pb':
        output_op_name = ["occupancy/conv2d/Sigmoid",
                          "loc/reshape/Reshape",
                          "size/reshape/Reshape",
                          "angle/conv2d/BiasAdd",
                          "heading/conv2d/Sigmoid",
                          "clf/reshape/Reshape", ]
    elif pb == 'dlrm_post.onnx':
        output_op_name = ['92', ]
    elif pb == 'vilbert-multi-task.onnx':
        output_op_name = ["vil_prediction",
                          "vil_prediction_gqa",
                          "vil_logit",
                          "vil_binary_prediction",
                          "vil_tri_prediction",
                          "vision_prediction",
                          "vision_logit",
                          "linguisic_prediction",
                          "linguisic_logit", ]
    elif pb == 'cycleGAN_net_G.onnx':
        output_op_name = ["185", ]
    else:
        return None
    return output_op_name


def get_inputs_dict(model):
    input_dict = {}

    if model == 'NCF_frozen_graph.pb':
        input_dict = {"user_input": [1, 1], "item_input": [1, 1]}
    elif model == 'alexnet_frozen_graph.pb':
        input_dict = {"input": [1, 227, 227, 3]}
    elif model == 'cosface_frozen_graph.pb':
        input_dict = {"image": [1, 112, 96, 3]}
    elif model == 'deeplab_v3_frozen_graph.pb':
        input_dict = {"ImageTensor": [1, 112, 112, 3]}
    elif model == 'densenet169_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'inception_v3_frozen_graph.pb':
        input_dict = {"input": [1, 299, 299, 3]}
    elif model == 'MaskRCNN_frozen_graph.pb':
        assert 0
    elif model == 'meta_bilstm_frozen_graph.pb':
        assert 0
    elif model == 'mobilenet_v1_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'mobilenet_v2_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'nin_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'R2CNN_resnet_v1_frozen_graph.pb':
        assert 0
    elif model == 'resnext101_32x4d_bottleneck_type_b_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'resnext101_32x4d_bottleneck_type_c_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'resnext50_32x4d_bottleneck_type_b_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'resnext50_32x4d_bottleneck_type_c_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'shufflenet_v1_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'simple_network_frozen_graph.pb':
        assert 0
    elif model == 'squeezenet_v1_frozen_graph.pb':
        input_dict = {"input": [1, 227, 227, 3]}
    elif model == 'stn_frozen_graph.pb':
        input_dict = {"input": [1, 42, 42, 1]}
    elif model == 'vgg16_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'xception_frozen_graph.pb':
        input_dict = {"input": [1, 299, 299, 3]}
    elif model == 'resnext101_32x4d_type_c_group_conv_frozen_graph_without_training.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'resnet_v1_101_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'resnet_v1_152_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'cosface_origin_tu_fp16_fold_constants.pb':
        input_dict = {"image": [1, 112, 96, 3]}
    elif model == 'Mixnet_L_frozen_graph_no_swish_f32.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'mobilenext_without_training_frozen_graph.pb':
        input_dict = {"input": [1, 224, 224, 3]}
    elif model == 'yolov3-10.onnx':
        input_dict = {'input_1': [1, 3, 416, 416], 'image_shape': [1, 2]}
    elif model == 'gpt2_fp32.onnx':
        input_dict = {'0': [1, 30]}
    elif model == 'gpt2_fp16.onnx':
        input_dict = {'0': [1, 30]}
    elif model == 'hifi_gan.onnx':
        input_dict = {'input': [1, 80, 142]}
    elif model == 'hifinet.onnx':
        input_dict = {'input': [1, 80, 352]}
    elif model == 'PointPillars_opt.pb':
        input_dict = {'pillars/input': [1, 12000, 100, 7],
                      'pillars/indices': [1, 12000, 3], }
    elif model == 'dlrm_post.onnx':
        input_dict = {'input.1': [128, 13],
                      'ly': [128, 1664], }
    elif model == 'vilbert-multi-task.onnx':
        input_dict = {"question": [2, 256],
                      "features": [2, 306, 2048],
                      "spatials": [2, 306, 5],
                      "segment_ids": [2, 256],
                      "input_mask": [2, 256],
                      "image_mask": [2, 306],
                      "task_tokens": [2, 1], }
    elif model == 'cycleGAN_net_G.onnx':
        input_dict = {"input.1": [1, 3, 256, 256], }
    elif model == 'vgg19_frozen_graph.pb':
        input_dict = {'input_1': [1, 224, 224, 3]}
    elif model == 'ssd_resnet34_frozen_graph.pb':
        input_dict = {'image': [1, 3, 1200, 1200]}
    elif model == 'ssd_resnet50_frozen_graph.pb':
        input_dict = {'image_tensor': [1, 1, 1, 3]}
    elif model == 'resnet50_v1.pb':
        input_dict = {'input_tensor': [1, 224, 224, 3]}
    elif model == 'SFDDetector.onnx':
        input_dict = {'imgs': [1, 3, 400, 600]}
    elif model == 'Wav2Lip.onnx':
        input_dict = {'mel_batch': [1, 1, 80, 16],
                      'img_batch': [1, 6, 96, 96], }
    elif model == 'clip_vit_new_fp16.onnx':
        input_dict = {'input' : [1, 3, 224, 224]}
    elif model == 'wide_and_deep_origin_tf23.pb':
        input_dict = {'doc_event_days_since_published_log_01scaled': [1, 1],
                      'doc_ad_days_since_published_log_01scaled': [1, 1],
                      'doc_event_doc_ad_sim_categories': [1, 1],
                      'doc_event_doc_ad_sim_topics': [1, 1],
                      'doc_event_doc_ad_sim_entities': [1, 1],
                      'pop_document_id': [1, 1],
                      'pop_publisher_id': [1, 1],
                      'pop_source_id': [1, 1],
                      'pop_ad_id': [1, 1],
                      'pop_advertiser_id': [1, 1],
                      'pop_campain_id': [1, 1],
                      'doc_views_log_01scaled': [1, 1],
                      'ad_views_log_01scaled': [1, 1],
                      'ad_id': [1, 1],
                      'event_platform': [1, 1],
                      'doc_id': [1, 1],
                      'doc_event_source_id': [1, 1],
                      'doc_event_publisher_id': [1, 1],
                      'event_geo_location': [1, 1],
                      'event_country': [1, 1],
                      'event_country_state': [1, 1],
                      'ad_advertiser': [1, 1],
                      'campaign_id': [1, 1],
                      'doc_ad_publisher_id': [1, 1],
                      'doc_event_id': [1, 1],
                      'doc_ad_source_id': [1, 1]
                     }
    elif model == 'slowfast_8x8_r50.onnx':
      input_dict = {"input.1":[1 ,3 ,8 ,256 ,455], "input.7":[1 ,3 ,32 ,256 ,455]}
    return input_dict


@pytest.mark.parametrize("model", models_list)
def test_network(model, model_dir, exec_batch, max_batch, out_node, weight_share, loop, random_input, slz, deslz, use_tvm_plugin, builder_flag):
    if use_tvm_plugin == "True":
        import dleol
    elif use_tvm_plugin == "False":
        pass
    else:
        raise Exception

    suffix = Path(model).suffix.lower()
    if suffix.startswith('.'):
        suffix = suffix[1:]

    if out_node == "":
        out_node = get_out_node(model)
    else:
        out_node = out_node.split(';')
    inputs_dict = get_inputs_dict(model)
    if model_dir == "":
        model_dir = frozen_pb_dir

    model, exec_batch, max_batch, outputs_dict, weight_share, builder_flag, random_input = ParamParser(model, model_dir,
                                                                             exec_batch, max_batch, out_node,
                                                                             weight_share, builder_flag, random_input)
    if suffix == 'onnx':
        model = extract_onnx(model, outputs_dict.keys())

    if deslz == "":
        engine = run_tc_build(model, max_batch, weight_share, builder_flag, inputs_dict, outputs_dict)
    else:
        with open(deslz, 'rb') as f:
            engine = nne.deserialize(f.read())

    if engine is None:
        raise AssertionError("Build engine failed")

    if slz != "":
        with open(slz, 'wb') as f:
            f.write(engine.serialize())

    context, bindings, input_bindings, output_bindings = run_nne_exec(engine, exec_batch, weight_share, random_input)

    # run golden
    if suffix == 'pb':
        res = run_tf(model, input_bindings, output_bindings, exec_batch)
    elif suffix == 'onnx':
        res = run_onnx(model, input_bindings, output_bindings, exec_batch)
    elif suffix == 'tflite':
        res = run_tflite(model, input_bindings, output_bindings, exec_batch)
        rtol = 0
        atol = 1
    else:
        AssertionError("Not Support")

    cost_time = 0.0
    for round in range(int(loop)):
        # run dli
        start_time = time.time()
        context.execute(exec_batch, [binding.mem.as_buffer(binding.size) for binding in bindings])
        end_time = time.time()
        cost_time += end_time - start_time
        print("run round {} cost {}, ".format(round, (end_time - start_time)*1000))
        for idx in range(len(output_bindings)):
            binding_shape = output_bindings[idx].shape
            output_shape = (binding_shape[0] * exec_batch,) + binding_shape[1:]
            output_bindings[idx].data = cuda.from_device(output_bindings[idx].mem, output_shape, output_bindings[idx].dtype)
        # verify result
        rtol = 1e-2
        atol = 1e-2
        if 'shanmaDetBatch1_v9_1.8.0_test.onnx' in model:
            rtol = 1e-1
            atol = 1e-1
        pass_outnode = 0
        for idx in range(len(output_bindings)):
            if(compare_result(output_bindings[idx].data, res[idx], output_bindings[idx].dtype, rtol=rtol, atol=atol) == False):
                print('output: {} mismatch'.format(output_bindings[idx].name))
            else:
                pass_outnode = pass_outnode + 1
                print('output: {} pass'.format(output_bindings[idx].name))
        if pass_outnode == len(output_bindings):
            print("PASS")
        else:
            raise AssertionError("Compare failed")
        print('Run Loop: {:d}, Pass'.format(round))

    print("run {} times, ".format(int(loop)))
    loop_count = int(loop)
    if int(loop) < 1:
        loop_count = 1

    print('avg time : {} ms'.format(cost_time*1000/loop_count))

@pytest.mark.parametrize("model", models_list)
def test_network_build(model, model_dir, exec_batch, max_batch, out_node, weight_share, loop, random_input, slz, deslz, use_tvm_plugin, builder_flag):
    if use_tvm_plugin == "True":
        import dleol
    elif use_tvm_plugin == "False":
        pass
    else:
        raise Exception

    suffix = Path(model).suffix.lower()
    if suffix.startswith('.'):
        suffix = suffix[1:]

    if out_node == "":
        out_node = get_out_node(model)
    else:
        out_node = out_node.split(';')
    inputs_dict = get_inputs_dict(model)
    if model_dir == "":
        model_dir = frozen_pb_dir

    model, exec_batch, max_batch, outputs_dict, weight_share, builder_flag, random_input = ParamParser(model, model_dir,
                                                                                                       exec_batch, max_batch, out_node,
                                                                                                       weight_share, builder_flag, random_input)
    if suffix == 'onnx':
        model = extract_onnx(model, outputs_dict.keys())

    engine = run_tc_build(model, max_batch, weight_share, builder_flag, inputs_dict, outputs_dict)

    if engine is None:
        raise AssertionError("Build engine failed")

@pytest.mark.parametrize("model", models_list)
def test_network_serialize(model, model_dir, exec_batch, max_batch, out_node, weight_share, file_name, builder_flag, random_input):
    if out_node == "":
        out_node = get_out_node(model)
    else:
        out_node = out_node.split(';')
    inputs_dict = get_inputs_dict(model)
    if model_dir == "":
        model_dir = frozen_pb_dir

    model, exec_batch, max_batch, outputs_dict, weight_share, builder_flag, random_input = ParamParser(model,
                                                                           model_dir, exec_batch, max_batch, out_node,
                                                                           weight_share,builder_flag)

    engine = run_tc_build(model, max_batch, weight_share, builder_flag, inputs_dict, outputs_dict)

    if engine is None:
        raise AssertionError("Build engine failed")

    with open(file_name, 'wb') as f:
        f.write(engine.serialize())

def test_network_deserialize(file_name, exec_batch, weight_share, random_input=0):
    if file_name == "":
        raise AssertionError("Please specify serialized file")
    if exec_batch == "":
        batch_size = 1
    else:
        try:
            batch_size = int(exec_batch)
            if batch_size < 1 or batch_size > 64:
                batch_size = 1
        except ValueError:
            batch_size = 1

    if weight_share not in weight_share_configs:
        weight_share = "0"
        print("not set weight share, use default weight in cluster 0")
    with open(file_name, 'rb') as f:
        engine = nne.deserialize(f.read())
        run_nne_exec(engine, batch_size, weight_share_configs[weight_share], random_input)

if __name__ == "__main__":
    test_network("resnet_v1_50_frozen_graph.pb")

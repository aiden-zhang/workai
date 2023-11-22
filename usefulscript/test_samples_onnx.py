import dlnne as nne
import numpy as np
import onnx
import onnx_extractor
import onnxruntime
import onnxruntime.backend
import os
import pytest
import pycuda.driver as cuda
import pycuda.autoinit
import random
import tvm
from tvm import relay


MODEL_DIR = os.environ.get("DL_MODEL_DIR", "NONE")

class Binding:
    def __init__(self, mem, size, shape, dtype):
        self.mem = mem
        self.size = size
        self.shape = shape
        self.dtype = dtype


def get_dtype(type):
    if type == nne.DataType.FLOAT:
        dtype = np.float32
    elif type == nne.DataType.HALF:
        dtype = np.float16
    elif type == nne.DataType.UINT8:
        dtype = np.uint8
    elif type == nne.DataType.UINT16:
        dtype = np.uint16
    elif type == nne.DataType.UINT32:
        dtype = np.uint32
    elif type == nne.DataType.UINT64:
        dtype = np.uint64
    elif type == nne.DataType.INT8:
        dtype = np.int8
    elif type == nne.DataType.INT16:
        dtype = np.int16
    elif type == nne.DataType.INT32:
        dtype = np.int32
    elif type == nne.DataType.INT64:
        dtype = np.int64
    else:
        raise AssertionError("Unknown data type")
    return dtype

def check_relay_with_reference(relay_out, tf_out):
    if isinstance(relay_out, tvm.nd.NDArray):
        np.testing.assert_allclose(tf_out[0], relay_out.asnumpy(), rtol=1e-2, atol=1e-2)
    else:
        if not isinstance(tf_out, list):
            tf_out = [tf_out]
        for x, y in zip(tf_out, [r.asnumpy() for r in relay_out]):
            np.testing.assert_allclose(x, y, rtol=1e-2, atol=1e-2)


def compare_with_onnx(test_result, ref_result, rtol, atol):
    test_result_reshape = test_result.reshape(-1)
    ref_result_reshape = ref_result.reshape(-1)
    # print("test result: \n{}\n".format(test_result_reshape))
    # print("ref result: \n{}\n".format(ref_result_reshape))
    print("compare with onnx:")
    is_same = np.isclose(test_result_reshape, ref_result_reshape, rtol=rtol, atol=atol)

    if not np.all(is_same):
        mismatch = 100.0 * (1 - np.count_nonzero(is_same) / is_same.size)
        print('Mismatch: {:.3g}%'.format(mismatch))
        diff = abs(test_result - ref_result)
        print("max diff: {}".format(diff.max()))
        nan = np.isnan(diff)
        if diff.max() > atol or ((nan == 1).any()):
            diff_cnt = diff[np.where(diff > atol)].size
            print('diff count: {}'.format(diff_cnt))
            log_cnt = min(20, diff_cnt)
            diff_index_log = np.argwhere(diff > atol)[:log_cnt]
            diff_idx = np.where(diff > atol)
            diff_value_log = diff[diff_idx][:log_cnt]
            test_result_log = test_result[diff_idx][:log_cnt]
            ref_result_log = ref_result[diff_idx][:log_cnt]
            for idx in range(log_cnt):
                print("diff index: {}".format(diff_index_log[idx]))
                print("nne_result:%9f, onnx_result:%9f, diff:%9f" % (test_result_log[idx], ref_result_log[idx], diff_value_log[idx]))
            raise AssertionError("Compare failed")
        else:
            print("PASS")
    else:
        print("PASS")


def get_output_names(model):
    if model == "inception-v1-9":
        outputs = ["prob_1"]
    elif model == "resnet18-v1-7":
        outputs = ["resnetv15_dense0_fwd"]
    elif model == "resnet50-caffe2-v1-9":
        outputs = ["gpu_0/softmax_1"]
    elif model == "squeezenet1.1-7":
        outputs = ["squeezenet0_flatten0_reshape0"]
    elif model == "mobilenetv2-7":
        outputs = ["mobilenetv20_output_flatten0_reshape0"]
    elif model == "yolov2-coco-9":
        outputs = ["218"]
    elif model == "squeezenet1.0-9":
        outputs = ["softmaxout_1"]
    elif model == "vgg16-bn-7":
        outputs = ["vgg0_dense2_fwd"]
    elif model == "densenet-9":
        outputs = ["fc6_1"]
    elif model == "densenet201_224_224_without_trainning":
        outputs = ["output"]
    elif model == "inceptionv3_224_224":
        outputs = ["output"]
    elif model == "mobilenet_224_224":
        outputs = ["output"]
    elif model == "segnet_224_224_without_trainning":
        outputs = ["output"]
    elif model == "yolov3-416_416":
        outputs = ["convolution_output2", "convolution_output1", "convolution_output"]
    elif model == "yolov3-10":
        outputs = ["yolonms_layer_1/ExpandDims_1:0", "yolonms_layer_1/ExpandDims_3:0",
                   "yolonms_layer_1/concat_2:0"]
    elif model == "senet-50_224_224":
        outputs = ["975"]
    elif model == "yolov3_tiny_1088_1920":
        outputs = []
    elif model == "yolov3_tiny_416_416":
        outputs = []
    elif model == "yolov3_tiny_544_960":
        outputs = []
    elif model == "yolov3_tiny_608_608":
        outputs = []
    elif model == "yolov3_tiny_736_1280":
        outputs = []
    elif model == "simple_addition":
        outputs = []

    return outputs


def extract_onnx(origin_onnx_model_path, input_names = None, output_names = None):
    """
    extract a sub onnx model from the origin onnx model
    :param origin_onnx_model_path: the path to the origin onnx model
    :param output_names: the check points
    :return: the path of the extracted sub onnx model
    """
    if output_names is None:
        output_names = []
    model = onnx.load(origin_onnx_model_path)
    params = [init_tensor.name for init_tensor in model.graph.initializer]
    shape_dict = {}

    if len(shape_dict) == 0:
        for i in model.graph.input:
            name = i.name
            if name in params:
                continue
            proto_shape = i.type.tensor_type.shape.dim
            shape = [d.dim_value if d.dim_param == "" else relay.Any() for d in proto_shape]
            shape_dict[name] = shape

    if len(input_names) == 0:
        input_names = []
        for k in shape_dict:
            input_names.append(k)

    if len(output_names) == 0:
        output_names = []
        for i in model.graph.output:
            output_names.append(i.name)

    print("check point: ", output_names)
    extractor = onnx_extractor.Extractor(model)
    extracted = extractor.extract_model(input_names, output_names)

    if extracted is None:
        onnx_save_path = origin_onnx_model_path
    else:
        onnx_save_path = '/tmp/onnx_network_tmp' + str(random.randint(10000, 20000)) + '.onnx'
        onnx.save(extracted, onnx_save_path)

    print("onnx_save_path: {}".format(onnx_save_path))

    return onnx_save_path


def run_onnx(model_path, input_names, output_names, feed_list):
    """
    run a specific onnx model, feed by feed_list,

    :param model_path: path to the onnx model
    :param input_names: the input names
    :param output_names: the output names
    :param feed_list: is an array with dim size 2, the first dim size is the batch size,
                      the second dim size is the number of inputs
    :return: the output of the model
    """
    real_onnx_path = extract_onnx(model_path, input_names, output_names)

    prepare = onnxruntime.backend.prepare(real_onnx_path, "CPU")

    for i, feed in enumerate(feed_list):
        ref = prepare.run(feed)
        if i == 0:
            output = ref
        else:
            for j in range(len(output_names)):
                output[j] = np.concatenate((output[j], ref[j]), axis=0)

    return output


def run_tc_onnx(model, batch_size, max_batch_size, weight_share_cfg, inputs_dict=None, outputs_dict=None):
    onnx_inputs_data = [[]] * batch_size
    tc_out = []
    bindings = []
    outputs = []
    batch_input_shape = []

    weight_mode = weight_share_cfg["weight_mode"]
    cluster_cfg = weight_share_cfg["cluster_cfg"]
    print("weight_mode: %s, cluster_cfg: %s" % (weight_mode, cluster_cfg))

    output_names = outputs_dict.keys()
    if model.endswith('.onnx') and (len(inputs_dict) != 0 or len(outputs_dict) != 0):
        model = extract_onnx(model, inputs_dict.keys(), outputs_dict.keys())

    with nne.Builder() as builder, nne.Parser() as parser:
        network = builder.create_network()

        for input_name in inputs_dict:
            parser.register_input(input_name, inputs_dict[input_name])
            print("register input: {}".format(input_name))

        for output_name in outputs_dict:
            parser.register_output(output_name)
            print("register output: {}".format(output_name))

        builder.config.ws_mode = weight_mode
        builder.config.max_batch_size = max_batch_size

        print("set max batch size to %d" % builder.config.max_batch_size)

        parser.parse(model, network)

        with builder.build_engine(network) as engine:
            context = engine.create_execution_context(cluster_cfg)
            num_bindings = engine.num_bindings
            print("num bindings = {}".format(num_bindings))

            for index in range(num_bindings):
                binding_name = engine.get_binding_name(index)

                if binding_name in inputs_dict and inputs_dict[binding_name] is not None:
                    binding_shape = inputs_dict[binding_name]
                elif binding_name in outputs_dict and outputs_dict[binding_name] is not None:
                    binding_shape = outputs_dict[binding_name]
                else:
                    binding_shape = engine.get_binding_shape(index)

                dtype = get_dtype(engine.get_binding_dtype(index))
                vol = 1

                for s in binding_shape:
                    vol *= s
                size = vol * dtype(1).nbytes

                input_shape = (binding_shape[0] * batch_size,) + binding_shape[1:]

                if engine.binding_is_input(index):
                    context.set_bindings_shape(index, binding_shape)
                    input_data = np.ones(input_shape).astype(dtype)
                    mem = cuda.to_device(input_data)
                    batch_inputs = np.split(input_data, batch_size)
                    for idx, onnx_input_data in enumerate(onnx_inputs_data):
                        onnx_input_data.append(batch_inputs[idx])
                else:
                    outputs.append(engine.get_binding_name(index))
                    mem = cuda.mem_alloc(size * batch_size)

                bindings.append(Binding(mem, size, binding_shape, dtype))
                batch_input_shape.append(input_shape)

            print("execute with batch %d" % batch_size)
            context.execute(batch_size, [binding.mem.as_buffer(binding.size) for binding in bindings])
            for index in range(engine.num_bindings):
                if not engine.binding_is_input(index):
                    tc_out.append(
                        cuda.from_device(bindings[index].mem, batch_input_shape[index], bindings[index].dtype))

    if model.endswith('.onnx'):
        onnx_out = run_onnx(model, inputs_dict.keys(), outputs, onnx_inputs_data)

        for i in range(len(tc_out)):
            compare_with_onnx(tc_out[i], onnx_out[i], 1e-2, 1e-2)
    else:
        print("not a onnx model, can not verify by onnxruntime!")


def run_tc_onnx_serialize(filename, model_abs_path, max_batch_size, model_name):
    output_names = get_output_names(model_name)
    outputs_dict = dict(zip(output_names, [None] * len(output_names)))
    weight_mode = nne.WeightShareMode.single
    with nne.Builder() as builder, nne.Parser() as parser:
        network = builder.create_network()
        for output_name in outputs_dict:
            parser.register_output(output_name)
            print("register output: {}".format(output_name))

        builder.config.ws_mode = weight_mode
        builder.config.max_batch_size = max_batch_size

        print("set max batch size to %d" % builder.config.max_batch_size)
        parser.parse(model_abs_path, network)
        with builder.build_engine(network) as engine:
            with open(filename, 'wb') as f:
                f.write(engine.serialize())


def run_tc_onnx_deserialize(filename, batch_size):
    onnx_inputs = []
    tc_out = []
    bindings = []
    batch_input_shape = []
    print("get batch size to %d" % batch_size)
    outputs = []
    onnx_inputs_data = [[]] * batch_size
    cluster_cfg = nne.ClusterConfig.cluser0
    with open(filename, 'rb') as f:
        engine = nne.deserialize(f.read())
        print("get max batch size to %d" % engine.max_batch_size)
        if(batch_size > engine.max_batch_size):
            raise AssertionError("max batch size must greater than execution batch size")
        context = engine.create_execution_context(cluster_cfg)
        num_bindings = engine.num_bindings
        print("num bindings", num_bindings)
        for index in range(num_bindings):
            binding_shape = engine.get_binding_shape(index)
            dtype = get_dtype(engine.get_binding_dtype(index))
            vol = 1

            for s in binding_shape:
                vol *= s
            size = vol * dtype(1).nbytes

            input_shape = (binding_shape[0] * batch_size,) + binding_shape[1:]

            if engine.binding_is_input(index):
                input_data = np.ones(input_shape).astype(dtype)
                mem = cuda.to_device(input_data)
                batch_inputs = np.split(input_data, batch_size)
                for idx, onnx_input_data in enumerate(onnx_inputs_data):
                    onnx_input_data.append(batch_inputs[idx])
            else:
                outputs.append(engine.get_binding_name(index))
                mem = cuda.mem_alloc(size * batch_size)

            bindings.append(Binding(mem, size, binding_shape, dtype))
            batch_input_shape.append(input_shape)

        print("execute with batch %d" % batch_size)
        context.execute(batch_size, [binding.mem.as_buffer(binding.size) for binding in bindings])
        for index in range(engine.num_bindings):
            if not engine.binding_is_input(index):
                tc_out.append(
                    cuda.from_device(bindings[index].mem, batch_input_shape[index], bindings[index].dtype))

onnx_models_file_map = {
    "densenet201_224_224_without_trainning": "densenet201_224_224_without_trainning.onnx",
    "densenet-9": "densenet-9.onnx",
    "inception-v1-9": "inception-v1-9.onnx",
    "inceptionv3_224_224": "inceptionv3_224_224.onnx",
    "mobilenetv2-7": "mobilenetv2-7.onnx",
    "mobilenet_224_224": "mobilenet_224_224.onnx",
    "resnet18-v1-7": "resnet18-v1-7.onnx",
    "resnet50-caffe2-v1-9": "resnet50-caffe2-v1-9.onnx",
    "segnet_224_224_without_trainning": "segnet_without_Aten.onnx",
    "senet-50_224_224": "senet-50_224_224.onnx",
    "squeezenet1.0-9": "squeezenet1.0-9.onnx",
    "squeezenet1.1-7": "squeezenet1.1-7.onnx",
    "vgg16-bn-7": "vgg16-bn-7.onnx",
    "yolov2-coco-9": "yolov2-coco-9.onnx",
    "yolov3-416_416": "yolov3-416_416.onnx",
    "yolov3-10": "yolov3-10.onnx",
    "yolov3_tiny_1088_1920": "yolov3_tiny_1088_1920.onnx",
    "yolov3_tiny_416_416": "yolov3_tiny_416_416.onnx",
    "yolov3_tiny_544_960": "yolov3_tiny_544_960.onnx",
    "yolov3_tiny_608_608": "yolov3_tiny_608_608.onnx",
    "yolov3_tiny_736_1280": "yolov3_tiny_736_1280.onnx",
    "simple_addition":"simple_addition.onnx"
}

weight_share_configs = {
    "0": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser0,
    },
    "1": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser1,
    },
    "2": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser2,
    },
    "3": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser3,
    },
    "01": {
        "weight_mode": nne.WeightShareMode.share2,
        "cluster_cfg": nne.ClusterConfig.cluser01,
    },
    "23": {
        "weight_mode": nne.WeightShareMode.share2,
        "cluster_cfg": nne.ClusterConfig.cluser23,
    },
    "0123": {
        "weight_mode": nne.WeightShareMode.share4,
        "cluster_cfg": nne.ClusterConfig.cluser0123,
    },
}


@pytest.mark.parametrize("model_name", onnx_models_file_map.keys())
def test_onnx_build_engine(model_name, model_dir, exec_batch, max_batch, out_node, weight_share):
    if exec_batch == "":
        batch_size = 1
    else:
        try:
            batch_size = int(exec_batch)
            if batch_size < 1 or batch_size > 64:
                batch_size = 1
        except ValueError:
            batch_size = 1

    if max_batch == "":
        max_batch_size = 1
    else:
        try:
            max_batch_size = int(max_batch)
            if max_batch_size < 1 or max_batch_size > 64:
                max_batch_size = 1
        except ValueError:
            max_batch_size = 1

    if max_batch_size < batch_size:
        raise AssertionError("max batch size must greater than execution batch size")

    if weight_share not in weight_share_configs:
        raise AssertionError("weight_share: %s is not a valid input" % weight_share)

    print("model_name=%s, batch_size=%d, max_batch_size=%d, outnode=%s"
          % (model_name, batch_size, max_batch_size, out_node))

    inputs_dict = {}
    outputs_dict = {}

    if out_node:
        # only work in 1 out_node case
        outputs_dict[out_node] = None
    else:
        output_names = get_output_names(model_name)
        outputs_dict = dict(zip(output_names, [None] * len(output_names)))

    if model_dir == "":
        raise AssertionError("Please specify model directory.")

    if model_dir.endswith('/'):
        model_abs_path = model_dir + onnx_models_file_map[model_name]
    else:
        model_abs_path = model_dir + '/' + onnx_models_file_map[model_name]

    print('model path: ', model_abs_path)
    run_tc_onnx_build_engine(model_abs_path,
                batch_size, max_batch_size,
                weight_share_configs[weight_share],
                inputs_dict, outputs_dict)


def run_tc_onnx_build_engine(model, batch_size, max_batch_size, weight_share_cfg, inputs_dict={}, outputs_dict={}):
    onnx_inputs_data = [[]] * batch_size
    tc_out = []
    bindings = []
    outputs = []
    batch_input_shape = []

    weight_mode = weight_share_cfg["weight_mode"]
    cluster_cfg = weight_share_cfg["cluster_cfg"]
    print("weight_mode: %s, cluster_cfg: %s" % (weight_mode, cluster_cfg))


    output_names = outputs_dict.keys()

    model = extract_onnx(model, output_names)

    with nne.Builder() as builder, nne.Parser() as parser:
        network = builder.create_network()

        for input_name in inputs_dict:
            parser.register_input(input_name, inputs_dict[input_name])
            print("register input: {}".format(input_name))

        for output_name in outputs_dict:
            parser.register_output(output_name)
            print("register output: {}".format(output_name))

        builder.config.ws_mode = weight_mode
        builder.config.max_batch_size = max_batch_size
        print("set max batch size to %d" % builder.config.max_batch_size)

        parser.parse(model, network)

        engine = builder.build_engine(network)
        if engine == None:
            print('Build Engine FAILED')
        else:
            print('PASSED')


# @pytest.mark.parametrize("model_name", onnx_models_file_map.keys())
def test_onnx(model_name, model_path, exec_batch, max_batch, out_node, weight_share):
    # import pdb; pdb.set_trace()
    onnxruntime.set_default_logger_severity(3)
    if exec_batch == "":
        batch_size = 1
    else:
        try:
            batch_size = int(exec_batch)
            if batch_size < 1 or batch_size > 64:
                batch_size = 1
        except ValueError:
            batch_size = 1

    if max_batch == "":
        max_batch_size = 1
    else:
        try:
            max_batch_size = int(max_batch)
            if max_batch_size < 1 or max_batch_size > 64:
                max_batch_size = 1
        except ValueError:
            max_batch_size = 1

    if max_batch_size < batch_size:
        raise AssertionError("max batch size must greater than execution batch size")

    if weight_share not in weight_share_configs:
        raise AssertionError("weight_share: %s is not a valid input" % weight_share)

    print("model_name=%s, batch_size=%d, max_batch_size=%d, outnode=%s"
          % (model_name, batch_size, max_batch_size, out_node))

    inputs_dict = {}
    outputs_dict = {}

    if out_node:
        # only work in 1 out_node case
        outputs_dict[out_node] = None
    else:
        print('no out_node')
        # output_names = get_output_names(model_name)
        # outputs_dict = dict(zip(output_names, [None] * len(output_names)))

    # if model_name == "yolov3-416_416":
    #     inputs_dict = {'input_1': [1, 3, 416, 416], 'image_shape': [1, 2]}
    #
    # if model_name == "yolov3-10":
    #     inputs_dict = {'input_1': (1, 3, 64, 64), 'image_shape': (1, 2)}
    #     outputs_dict = {"yolonms_layer_1/ExpandDims_1:0": (1, 252, 4),
    #                     "yolonms_layer_1/ExpandDims_3:0": (1, 80, 252),
    #                     "yolonms_layer_1/concat_2:0": (0, 3)}

    if model_path == "":
        raise AssertionError("Please specify model directory.")

    # if model_dir.endswith('/'):
    #     model_abs_path = model_dir + onnx_models_file_map[model_name]
    # else:
    #     model_abs_path = model_dir + '/' + onnx_models_file_map[model_name]

    print('model path: ', model_path)
    run_tc_onnx(model_path,
                batch_size, max_batch_size,
                weight_share_configs[weight_share],
                inputs_dict, outputs_dict)

# @pytest.mark.parametrize("model_name", onnx_models_file_map.keys())
def test_onnx_serialize(file_name, model_name, max_batch, model_dir):
    if model_dir == "":
        raise AssertionError("Please specify model directory.")

    if model_dir.endswith('/'):
        model_abs_path = model_dir + onnx_models_file_map[model_name]
    else:
        model_abs_path = model_dir + '/' + onnx_models_file_map[model_name]
    if max_batch == "":
        max_batch_size = 1
    else:
        try:
            max_batch_size = int(max_batch)
            if max_batch_size < 1 or max_batch_size > 64:
                max_batch_size = 1
        except ValueError:
            max_batch_size = 1
    run_tc_onnx_serialize(file_name, model_abs_path, max_batch_size, model_name)


def test_onnx_deserialize(file_name, exec_batch):
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
    run_tc_onnx_deserialize(file_name, batch_size)

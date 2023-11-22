# This Python file uses the following encoding: utf-8

import dlnne as nne
import numpy as np
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit
import random
from tvm import relay
import onnx
import onnx_extractor
import onnxruntime
import onnxruntime.backend
import tempfile



if tf.__version__ == "1.12.0":
    from tensorflow.contrib.rnn import *


class Binding:
    def __init__(self, name, data, mem, size, shape, dtype, is_input):
        self.name = name
        self.mem = mem
        self.data = data
        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.input = is_input

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
    elif type == nne.DataType.BOOL:
        dtype = np.bool8
    else:
        raise AssertionError("Unknown data type")
    return dtype

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

builder_flag_map = {
    "spm_alloc": nne.BuilderFlag.spm_alloc,
}

def ParamParser(model, model_dir, exec_batch, max_batch, out_node, weight_share, builder_flag, random_input='2'):
    if exec_batch == "":
        batch_size = 1
    else:
        try:
            batch_size = int(exec_batch)
            if batch_size < 1:
                raise Exception('batch_size < 1 is illegal')
        except Exception as e:
            print(e)
            exit()

    if max_batch == "":
        max_batch_size = 1
    else:
        try:
            max_batch_size = int(max_batch)
            if max_batch_size < 1:
                raise Exception('max_batch_size < 1 is illegal')
        except Exception as e:
            print(e)
            exit()

    if max_batch_size < batch_size:
        raise AssertionError("max batch size must greater than execution batch size")

    if weight_share not in weight_share_configs:
        weight_share = "0"
        print("not set weight share, use default weight in cluster 0")

    flags = builder_flag.replace(' ', '').split('|') if builder_flag else []  # clear spaces and split flags with '|'
    builder_flag = 0
    for f in flags:
        if f not in builder_flag_map:
            raise AssertionError("'{}' is not a valid builder flag".format(f))
        builder_flag |= builder_flag_map[f]

    print("model_name=%s, batch_size=%d, max_batch_size=%d, outnode=%s, builder_flag=%#x"
          % (model, batch_size, max_batch_size, out_node, builder_flag))

    outputs_dict = {}

    if out_node:
        out_node_list = out_node[0].split(',')
        outputs_dict = dict(zip(out_node_list, [None] * len(out_node_list)))

    if model_dir == "":
        raise AssertionError("Please specify model directory.")

    if model_dir.endswith('/'):
        model_abs_path = model_dir + model
    else:
        model_abs_path = model_dir + '/' + model

    if random_input:
        if int(random_input) != 0 and int(random_input) != 1 and int(random_input) != 2:
            raise AssertionError("invalid random input option")

    return model_abs_path, batch_size, max_batch_size, outputs_dict, weight_share_configs[weight_share], builder_flag, int(random_input)

def extract_onnx(origin_onnx_model_path, output_names):
    """
    extract a sub onnx model from the origin onnx model
    :param origin_onnx_model_path: the path to the origin onnx model
    :param output_names: the check points
    :return: the path of the extracted sub onnx model
    """
    model = onnx.load(origin_onnx_model_path)
    params = [init_tensor.name for init_tensor in model.graph.initializer]
    input_names = []
    shape_dict = {}

    if len(shape_dict) == 0:
        for i in model.graph.input:
            name = i.name
            if name in params:
                continue
            proto_shape = i.type.tensor_type.shape.dim
            shape = [d.dim_value if d.dim_param == "" else relay.Any() for d in proto_shape]
            shape_dict[name] = shape

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
        onnx_save_path = gen_temp_file_name(suffix='.onnx')
        onnx.save(extracted, onnx_save_path)

    return onnx_save_path

def compare_result(test_result, ref_result, input_dtype, rtol, atol):
    print("compare result:")
    test_result = test_result.astype(input_dtype)
    ref_result = ref_result.astype(input_dtype)
    is_same = np.isclose(test_result, ref_result, rtol=rtol, atol=atol)

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
                print("nne_result:%9f, golden_result:%9f, diff:%9f" % (test_result_log[idx], ref_result_log[idx], diff_value_log[idx]))
        return False
    else:
        return True


def run_tc_build(model, max_batch_size, weight_share_cfg, builder_flag, inputs_dict={}, outputs_dict={}):
    weight_mode = weight_share_cfg["weight_mode"]
    cluster_cfg = weight_share_cfg["cluster_cfg"]
    print("weight_mode: %s, cluster_cfg: %s" % (weight_mode, cluster_cfg))
    print("inputs: ", inputs_dict)

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
        builder.set_flags(builder_flag)
        print("set max batch size to %d" % builder.config.max_batch_size)

        parser.parse(model, network)

        return builder.build_engine(network)

def run_nne_exec(engine, batch_size, weight_share_cfg, random_input=2):
    bindings = []
    outputs_bindings = []
    inputs_bindings = []
    cluster_cfg = weight_share_cfg["cluster_cfg"]
    context = engine.create_execution_context(cluster_cfg)
    num_bindings = engine.num_bindings
    print("num bindings = {}".format(num_bindings))

    for index in range(num_bindings):
        binding_shape = engine.get_binding_shape(index)
        dtype = get_dtype(engine.get_binding_dtype(index))
        vol = 1

        for s in binding_shape:
            vol *= s
        size = vol * dtype(1).nbytes

        input_shape = (binding_shape[0] * batch_size,) + binding_shape[1:]

        if engine.binding_is_input(index):
            if random_input == 0:
                input_data = np.ones(input_shape).astype(dtype)
            elif random_input == 1:
                np.random.seed(42)
                input_data = np.random.uniform(low=0.0, high=5.0, size=input_shape).astype(dtype)
            else:
                input_data = np.random.uniform(low=0.0, high=5.0, size=input_shape).astype(dtype)
            mem = cuda.to_device(input_data)
            binding = Binding(engine.get_binding_name(index), input_data, mem, size, binding_shape, dtype, True)
            bindings.append(binding)
            inputs_bindings.append(binding)
        else:
            mem = cuda.mem_alloc(size * batch_size)
            binding = Binding(engine.get_binding_name(index), None, mem, size, binding_shape, dtype, False)
            bindings.append(binding)
            outputs_bindings.append(binding)

    print("execute with batch %d" % batch_size)
    return context, bindings, inputs_bindings, outputs_bindings


def run_tc(model, batch_size, max_batch_size, weight_share_cfg, builder_flag, inputs_dict={}, outputs_dict={}):
   engine = run_tc_build(model, max_batch_size, weight_share_cfg, builder_flag, inputs_dict, outputs_dict)

   if engine is None:
       raise AssertionError("Build engine failed")

   return run_nne_exec(engine, batch_size, weight_share_cfg)

def gen_temp_file_name(prefix="", suffix=""):
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
    temp_file.close()
    return temp_file.name

# if__name__ == "__main__":
#     pass

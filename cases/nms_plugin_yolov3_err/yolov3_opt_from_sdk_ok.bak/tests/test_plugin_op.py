from __future__ import absolute_import, print_function
import tvm
from tvm import relay
import numpy as np
import dl
import pdb
from tvm.topi.utils import get_const_tuple
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay.frontend.common import infer_type as _infer_type
from tvm.relay.frontend.onnx import (
    OnnxOpConverter,
    _get_convert_map,
)
from tvm.target import override_native_generic_func
from tvm.relay.op.strategy.generic import wrap_topi_schedule
from tvm import te

original_get_convert_map = _get_convert_map


def to_numpy(out):
    if isinstance(out, tvm.nd.NDArray):
        # Single result
        return out.asnumpy()
    else:
        # Multiple results
        return [r.asnumpy() for r in out]

import sys
import os

KERNEL_PATH=os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH=os.path.join(KERNEL_PATH,"../dlnne_plugin/plugin/kernel")

if ""==os.getenv("YOLOV3_PLUGIN_KERNEL_PATH",""):
    os.environ["YOLOV3_PLUGIN_KERNEL_PATH"]=KERNEL_PATH

import dl
from dl import op
import pytest

BasePath=os.path.dirname(os.path.abspath(__file__))
plugin_tvm_so=os.path.join(BasePath,"../dlnne_plugin_build/libyolov3_opt_plugin.so")
TVM_TVM_REGISTER_SO_NAME = os.path.join(BasePath,"../dlnne_plugin_build/libyolov3_opt_tvm.so")
op.load_op_library(plugin_tvm_so)
op.load_op_library(TVM_TVM_REGISTER_SO_NAME)


front_end = os.path.join(BasePath,"../front_end.py")
import importlib.util
spec = importlib.util.spec_from_file_location("plugin_tvm", front_end)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
from tvm.relay import op as _op, expr as _expr



def evaluate_with_nne(mod, inputs_dict, config_key="0",serialize_file=None,deseralize=False):
    import dlnne as nne
    import tempfile
    import pycuda.driver as cuda
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

    if isinstance(mod["main"].body.checked_type, tvm.ir.type.TupleType):
        out_tensors = mod["main"].body.checked_type.fields
        outputs_shape_dict = {
            "out_{}".format(idx): get_const_tuple(tensor.shape)
            for idx, tensor in enumerate(out_tensors)
        }
        outputs_name = ["out_{}".format(idx) for idx in range(len(out_tensors))]

    else:
        out_shape = get_const_tuple(mod["main"].body.checked_type.shape)

        outputs_shape_dict = {"out_0": out_shape}
        outputs_name = ["out_0"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rlym") as f:
        f.write(tvm.ir.save_json(mod))
        f.flush()

        with nne.Builder() as builder, nne.Parser() as parser:
            parser.register_user_op(TVM_TVM_REGISTER_SO_NAME,front_end,"custom_op")

            weight_mode = weight_share_configs[config_key]["weight_mode"]
            cluster_cfg = weight_share_configs[config_key]["cluster_cfg"]
            print("weight_mode: %s, cluster_cfg: %s" % (weight_mode, cluster_cfg))

            network = builder.create_network()
            builder.config.ws_mode = weight_mode
            builder.config.max_batch_size = 1
            # network.num_inputs()
            # network.num_outputs()
            # [parser.register_output(key) for key, value in outputs_shape_dict.items()]
            if serialize_file != None and deseralize:

                engine = nne.deserialize(serialize_file.read())
            else:
                parser.parse(f.name, network)
                engine = builder.build_engine(network)

            nb = engine.num_bindings
            for i in np.arange(nb):
                print(f'name:{engine.get_binding_name(i)},shape:{engine.get_binding_shape(i)}')
            # builder.max_batch_size = 1
            context = engine.create_execution_context(cluster_cfg)

            batch_size = 1


            def _evaluate(inputs_dict):
                outputs = []
                keys = []
                bindings = []
                batch_input_shape = []
                for index in range(nb):
                    binding_name = engine.get_binding_name(index)

                    if binding_name in inputs_dict:
                        binding_shape = inputs_dict[binding_name].shape
                        context.set_bindings_shape(index, binding_shape)
                    else:
                        binding_shape = outputs_shape_dict[binding_name]

                    dtype = get_dtype(engine.get_binding_dtype(index))

                    vol = 1
                    for s in binding_shape:
                        vol *= s
                    size = 1
                    if dtype == np.bool:
                        size = vol*size
                    else:
                        size = vol * dtype(1).nbytes
                    input_shape = (binding_shape[0] * batch_size,) + binding_shape[1:]
                    if engine.binding_is_input(index):
                        mem = cuda.to_device(inputs_dict[binding_name])
                    else:
                        mem = cuda.mem_alloc(size * batch_size)
                    bindings.append(Binding(mem, size, binding_shape, dtype))
                    batch_input_shape.append(input_shape)

                context.execute(
                    batch_size, [binding.mem.as_buffer(binding.size) for binding in bindings]
                )

                for name in outputs_name:
                    index = engine.get_binding_index(name)
                    outputs.append(
                        cuda.from_device(
                            bindings[index].mem, batch_input_shape[index], bindings[index].dtype
                        )
                    )

                assert len(outputs)
                return outputs

            if len(outputs_shape_dict.keys()) == 1:
                rs= _evaluate(inputs_dict)[0]
            else:
                rs= _evaluate(inputs_dict)

            if serialize_file != None and deseralize is False:
                serialize_file.write(engine.serialize())

            return rs

class Binding:
    def __init__(self, mem, size, shape, dtype):
        self.mem = mem
        self.size = size
        self.shape = shape
        self.dtype = dtype


def get_dtype(type):
    import dlnne as nne

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
    elif type == nne.DataType.INT8:
        dtype = np.int8
    elif type == nne.DataType.INT16:
        dtype = np.int16
    elif type == nne.DataType.INT32:
        dtype = np.int32
    elif type == nne.DataType.INT64:
        dtype = np.int64
    elif type == nne.DataType.BOOL:
        dtype = np.bool
    else:
        raise AssertionError("Unknown data type")
    return dtype


# from testing import evaluate_with_nne
def run_mod(mod, params, runtime="vm", open_opt=False,serialize_file=None,deseralize=False):
    if open_opt:
        mod = dl.relay.ir_pass.optimize(mod)
        print(mod)

    if runtime in ["vm", "graph"]:
        for k, v in params.items():
            if isinstance(v, tvm.nd.NDArray) is False:
                params[k] = tvm.nd.array(v)

        ex = relay.create_executor(runtime, mod=mod, target="cuda")
        relay_out = ex.evaluate(mod["main"])(**params)

        if isinstance(relay_out, tvm.runtime.container.ADT):
            return [out.asnumpy() for out in relay_out]
        return relay_out.asnumpy()

    else:
        for k, v in params.items():
            if isinstance(v, tvm.nd.NDArray):
                params[k] = v.asnumpy()

        return evaluate_with_nne(mod, params,"0",serialize_file,deseralize)



def test_custom_csum_plugin(serialize_file=None,deseralize=False):
    count=relay.var("csum_in",shape=[1,80],dtype="int32")
    csum_value=tvm.get_global_func("dl.relay.op._make.custom_csum")(count, -1, 1)
    in_data=np.random.randint(0,2,size=[1,80]).astype("int32")
    params={
        "csum_in":in_data
    }

    func = relay.Function([count], csum_value)

    mod = tvm.IRModule()
    mod["main"] = func

    result = run_mod(mod, params.copy(), "nne", open_opt=True,
                     serialize_file=serialize_file,deseralize=deseralize)
    golden=np.cumsum(params["csum_in"],axis=-1).reshape([-1])

    print(golden,result)
    assert  np.allclose(result,golden)

    print("PASS")



def test_custom_combine_non_max_suppression_post(serialize_file=None,deseralize=False):
    import numpy as np
    in0=np.load(BasePath+"/nms_post_golden/input_0.npy")
    in1=np.load(BasePath+"/nms_post_golden/input_1.npy")
    in2=np.load(BasePath+"/nms_post_golden/input_2.npy")
    in3=np.load(BasePath+"/nms_post_golden/input_3.npy")
    in4=np.load(BasePath+"/nms_post_golden/input_4.npy")

    in0_var=relay.var("in0",shape=[1,80,10647,4],dtype="float32")
    in1_var=relay.var("in1",shape=[1,80,16384],dtype="float32")
    in2_var=relay.var("in2",shape=[1,80,10647],dtype="int32")
    in3_var=relay.var("in3",shape=[1,80],dtype="int32")
    in4_var=relay.var("in4",shape=[80],dtype="int32")

    post_func = tvm.get_global_func("dl.relay.op._make.custom_combine_non_max_suppression_post")
    values=post_func(in0_var,in1_var,in2_var,in3_var,in4_var,9999)

    func = relay.Function([in0_var,in1_var,in2_var,in3_var,in4_var], values)

    mod = tvm.IRModule()
    mod["main"]=func
    params={
        "in0":in0,
        "in1":in1,
        "in2":in2,
        "in3":in3,
        "in4":in4
    }
    result = run_mod(mod, params.copy(), "nne", open_opt=True,
                     serialize_file=serialize_file,deseralize=deseralize)

    target0=np.load(BasePath+"/nms_post_golden/target_0.npy")
    target1=np.load(BasePath+"/nms_post_golden/target_1.npy")
    target2=np.load(BasePath+"/nms_post_golden/target_2.npy")
    target3=np.load(BasePath+"/nms_post_golden/target_3.npy")

    assert  np.max(np.abs(result[3]-target3))==0
    valid_num=result[3][0]
    # print(result[1],target1)

    result_0=result[0][:valid_num]
    result_1=result[1][:valid_num]
    result_2=result[2][:valid_num]

    target0=target0[:valid_num]
    target1=target1[:valid_num]
    target2=target2[:valid_num]

    print(result_0,target0)
    print(result_1,target1)
    print(result_2,target2)

    print(result[3],target3)

    assert np.allclose(target0,result_0)
    # assert np.allclose(target1,result_1) #todo open it when nne fix
    assert np.allclose(target2,result_2)

    print("PASS")

def test_custom_filter_sort_416_416(serialize_file=None,deseralize=False):
    tu77 = relay.var("tu77", shape=[1, 3, 13, 13], dtype="float16")
    tu77_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu77.npy").astype(np.float16), (1, 3, 13, 13))
    tu78 = relay.var("tu78", shape=[1, 240, 13, 13], dtype="float16")
    tu78_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu78.npy").astype(np.float16), (1, 240, 13, 13))
    tu79 = relay.var("tu79", shape=[1, 3, 26, 26], dtype="float16")
    tu79_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu79.npy").astype(np.float16), (1, 3, 26, 26))
    tu80 = relay.var("tu80", shape=[1, 240, 26, 26], dtype="float16")
    tu80_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu80.npy").astype(np.float16), (1, 240, 26, 26))
    tu81 = relay.var("tu81", shape=[1, 3, 52, 52], dtype="float16")
    tu81_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu81.npy").astype(np.float16), (1, 3, 52, 52))
    tu82 = relay.var("tu82", shape=[1, 240, 52, 52], dtype="float16")
    tu82_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu82.npy").astype(np.float16), (1, 240, 52, 52))

    params = {
        "tu77": tu77_np,
        "tu78": tu78_np,
        "tu79": tu79_np,
        "tu80": tu80_np,
        "tu81": tu81_np,
        "tu82": tu82_np
    }

    custom_filter_sort = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        _expr.Tuple([tu77, tu78, tu79, tu80, tu81, tu82]),
        0.3,
        False,
        "float32",
        "int32",
        [[80]+[16384],[80]+[16384]],
        )

    func = relay.Function([tu77, tu78, tu79, tu80, tu81, tu82], custom_filter_sort);
    mod = tvm.IRModule()
    mod["main"] = func.with_attr("outputs", ["out_0", "out_1", "out_2"])

    result = run_mod(mod, params.copy(), "nne", open_opt=True,
                    serialize_file=serialize_file, deseralize=deseralize)

    out_0 = result[0]
    out_1 = result[1]
    out_2 = result[2]

    golden_0 = np.load(BasePath+"/filter_sort_golden/filter_sort_out0.npy")
    golden_1 = np.load(BasePath+"/filter_sort_golden/filter_sort_out1.npy")
    golden_2 = np.load(BasePath+"/filter_sort_golden/filter_sort_out2.npy")

    data_list = []
    golden_data_list = []
    idx_list = []
    golden_idx_list = []
    for i in range(80):
        if out_2[i]>0:
            total = out_2[i]
            for j in range(total):
                data_list.append(out_0[i, j])
                idx_list.append(out_1[i, j])
        if golden_2[i]>0:
            golden_total = golden_2[i]
            for k in range(golden_total):
                golden_data_list.append(golden_0[i, k])
                golden_idx_list.append(golden_1[i, k])
    
    print(data_list)
    print(idx_list) 
    assert np.allclose(np.array(data_list), np.array(golden_data_list), rtol=1.e-4, atol=1.e-3)
    assert np.allclose(np.array(idx_list), np.array(golden_idx_list), rtol=1.e-5, atol=1.e-6)
    assert np.allclose(golden_2, out_2, rtol=1.e-5, atol=1.e-6)
    print("==================================== PASS =========================================")

def test_custom_filter_sort_pure_416_416(serialize_file=None,deseralize=False):
    tu77_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu77.npy").astype(np.float16), (1, 3, 13, 13))
    tu78_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu78.npy").astype(np.float16), (1, 240, 13, 13))
    reshape_1_1 = np.reshape(tu77_np,[1,3,-1])
    reshape_1_2 = np.reshape(tu78_np,[80,3,-1])
    mul_1 = reshape_1_1 * reshape_1_2
    transpose_1 = np.transpose(mul_1,[0,2,1])
    reshape_1 = np.reshape(transpose_1,[1,80,-1])

    tu79_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu79.npy").astype(np.float16), (1, 3, 26, 26))
    tu80_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu80.npy").astype(np.float16), (1, 240, 26, 26))
    reshape_2_1 = np.reshape(tu79_np,[1,3,-1])
    reshape_2_2 = np.reshape(tu80_np,[80,3,-1])
    mul_2 = reshape_2_1 * reshape_2_2
    transpose_2 = np.transpose(mul_2,[0,2,1])
    reshape_2 = np.reshape(transpose_2,[1,80,-1])

    tu81_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu81.npy").astype(np.float16), (1, 3, 52, 52))
    tu82_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu82.npy").astype(np.float16), (1, 240, 52, 52))
    reshape_3_1 = np.reshape(tu81_np,[1,3,-1])
    reshape_3_2 = np.reshape(tu82_np,[80,3,-1])
    mul_3 = reshape_3_1 * reshape_3_2
    transpose_3 = np.transpose(mul_3,[0,2,1])
    reshape_3 = np.reshape(transpose_3,[1,80,-1])

    concat = np.concatenate([reshape_1,reshape_2,reshape_3],2)
    cast_np = concat.astype("float32")
    cast_np = np.reshape(cast_np,[80,10647])
    print("cast_np.shape: ",cast_np.shape)
    cast = relay.var("cast", shape=[80,10647], dtype="float32")
    
    params = {
        "cast": cast_np,
    }

    custom_filter_sort = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        _expr.Tuple([cast]),
        0.3,
        False,
        "float32",
        "int32",
        [[80]+[16384],[80]+[16384]],
        )

    func = relay.Function([cast], custom_filter_sort);
    mod = tvm.IRModule()
    mod["main"] = func.with_attr("outputs", ["out_0", "out_1", "out_2"])

    result = run_mod(mod, params.copy(), "nne", open_opt=True,
                    serialize_file=serialize_file, deseralize=deseralize)

    out_0 = result[0]
    out_1 = result[1]
    out_2 = result[2]

    golden_0 = np.load(BasePath+"/filter_sort_golden/filter_sort_out0.npy")
    golden_1 = np.load(BasePath+"/filter_sort_golden/filter_sort_out1.npy")
    golden_2 = np.load(BasePath+"/filter_sort_golden/filter_sort_out2.npy")

    data_list = []
    golden_data_list = []
    idx_list = []
    golden_idx_list = []
    for i in range(80):
        if out_2[i]>0:
            total = out_2[i]
            for j in range(total):
                data_list.append(out_0[i, j])
                idx_list.append(out_1[i, j])
        if golden_2[i]>0:
            golden_total = golden_2[i]
            for k in range(golden_total):
                golden_data_list.append(golden_0[i, k])
                golden_idx_list.append(golden_1[i, k])
    
    print(data_list)
    print(idx_list)
    
    assert np.allclose(np.array(data_list), np.array(golden_data_list), rtol=1.e-4, atol=1.e-3)
    assert np.allclose(np.array(idx_list), np.array(golden_idx_list), rtol=1.e-5, atol=1.e-6)
    assert np.allclose(golden_2, out_2, rtol=1.e-5, atol=1.e-6)
    print("==================================== PASS =========================================")

def test_custom_filter_sort_720p(serialize_file=None,deseralize=False):
    tu94 = relay.var("tu94", shape=[1, 3, 23, 40], dtype="float16")
    tu94_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu94.npy").astype(np.float16), (1, 3, 23, 40))
    tu95 = relay.var("tu95", shape=[1, 240, 23, 40], dtype="float16")
    tu95_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu95.npy").astype(np.float16), (1, 240, 23, 40))
    tu96 = relay.var("tu96", shape=[1, 3, 46, 80], dtype="float16")
    tu96_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu96.npy").astype(np.float16), (1, 3, 46, 80))
    tu97 = relay.var("tu97", shape=[1, 240, 46, 80], dtype="float16")
    tu97_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu97.npy").astype(np.float16), (1, 240, 46, 80))
    tu98 = relay.var("tu98", shape=[1, 3, 92, 160], dtype="float16")
    tu98_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu98.npy").astype(np.float16), (1, 3, 92, 160))
    tu99 = relay.var("tu99", shape=[1, 240, 92, 160], dtype="float16")
    tu99_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu99.npy").astype(np.float16), (1, 240, 92, 160))

    params = {
        "tu94": tu94_np,
        "tu95": tu95_np,
        "tu96": tu96_np,
        "tu97": tu97_np,
        "tu98": tu98_np,
        "tu99": tu99_np
    }

    custom_filter_sort = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        _expr.Tuple([tu94, tu95, tu96, tu97, tu98, tu99]),
        0.3,
        False,
        "float32",
        "int32",
        [[80]+[65536],[80]+[65536]],
        )

    func = relay.Function([tu94, tu95, tu96, tu97, tu98, tu99], custom_filter_sort);
    mod = tvm.IRModule()
    mod["main"] = func.with_attr("outputs", ["out_0", "out_1", "out_2"])

    result = run_mod(mod, params.copy(), "nne", open_opt=True,
                    serialize_file=serialize_file, deseralize=deseralize)

    out_0 = result[0]
    out_1 = result[1]
    out_2 = result[2]

    golden_0 = np.load(BasePath+"/filter_sort_golden/filter_sort_720p_out0.npy")
    golden_1 = np.load(BasePath+"/filter_sort_golden/filter_sort_720p_out1.npy")
    golden_2 = np.load(BasePath+"/filter_sort_golden/filter_sort_720p_out2.npy")

    data_list = []
    golden_data_list = []
    idx_list = []
    golden_idx_list = []
    for i in range(80):
        if out_2[i]>0:
            total = out_2[i]
            for j in range(total):
                data_list.append(out_0[i, j])
                idx_list.append(out_1[i, j])
        if golden_2[i]>0:
            golden_total = golden_2[i]
            for k in range(golden_total):
                golden_data_list.append(golden_0[i, k])
                golden_idx_list.append(golden_1[i, k])
    
    print(data_list)
    print(idx_list) 
    assert np.allclose(np.array(data_list), np.array(golden_data_list), rtol=1.e-4, atol=1.e-3)
    assert np.allclose(np.array(idx_list), np.array(golden_idx_list), rtol=1.e-5, atol=1.e-6)
    assert np.allclose(golden_2, out_2, rtol=1.e-5, atol=1.e-6)
    print("==================================== PASS =========================================")

def test_custom_filter_sort_pure_720p(serialize_file=None,deseralize=False):
    tu94_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu94.npy").astype(np.float16), (1, 3, 23, 40))
    tu95_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu95.npy").astype(np.float16), (1, 240, 23, 40))
    reshape_1_1 = np.reshape(tu94_np,[1,3,-1])
    reshape_1_2 = np.reshape(tu95_np,[80,3,-1])
    mul_1 = reshape_1_1 * reshape_1_2
    transpose_1 = np.transpose(mul_1,[0,2,1])
    reshape_1 = np.reshape(transpose_1,[1,80,-1])

    tu96_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu96.npy").astype(np.float16), (1, 3, 46, 80))
    tu97_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu97.npy").astype(np.float16), (1, 240, 46, 80))
    reshape_2_1 = np.reshape(tu96_np,[1,3,-1])
    reshape_2_2 = np.reshape(tu97_np,[80,3,-1])
    mul_2 = reshape_2_1 * reshape_2_2
    transpose_2 = np.transpose(mul_2,[0,2,1])
    reshape_2 = np.reshape(transpose_2,[1,80,-1])

    tu98_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu98.npy").astype(np.float16), (1, 3, 92, 160))
    tu99_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu99.npy").astype(np.float16), (1, 240, 92, 160))
    reshape_3_1 = np.reshape(tu98_np,[1,3,-1])
    reshape_3_2 = np.reshape(tu99_np,[80,3,-1])
    mul_3 = reshape_3_1 * reshape_3_2
    transpose_3 = np.transpose(mul_3,[0,2,1])
    reshape_3 = np.reshape(transpose_3,[1,80,-1])

    concat = np.concatenate([reshape_1,reshape_2,reshape_3],2)
    cast_np = concat.astype("float32")
    cast_np = np.reshape(cast_np,[80,57960])
    print("cast_np.shape: ",cast_np.shape)

    cast = relay.var("cast", shape=[80,57960], dtype="float32")

    params = {
        "cast": cast_np,
    }

    custom_filter_sort = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        _expr.Tuple([cast]),
        0.3,
        False,
        "float32",
        "int32",
        [[80]+[65536],[80]+[65536]],
        )

    func = relay.Function([cast], custom_filter_sort);
    mod = tvm.IRModule()
    mod["main"] = func.with_attr("outputs", ["out_0", "out_1", "out_2"])

    result = run_mod(mod, params.copy(), "nne", open_opt=True,
                    serialize_file=serialize_file, deseralize=deseralize)

    out_0 = result[0]
    out_1 = result[1]
    out_2 = result[2]

    golden_0 = np.load(BasePath+"/filter_sort_golden/filter_sort_720p_out0.npy")
    golden_1 = np.load(BasePath+"/filter_sort_golden/filter_sort_720p_out1.npy")
    golden_2 = np.load(BasePath+"/filter_sort_golden/filter_sort_720p_out2.npy")

    data_list = []
    golden_data_list = []
    idx_list = []
    golden_idx_list = []
    for i in range(80):
        if out_2[i]>0:
            total = out_2[i]
            for j in range(total):
                data_list.append(out_0[i, j])
                idx_list.append(out_1[i, j])
        if golden_2[i]>0:
            golden_total = golden_2[i]
            for k in range(golden_total):
                golden_data_list.append(golden_0[i, k])
                golden_idx_list.append(golden_1[i, k])
    
    print(data_list)
    print(idx_list) 
    assert np.allclose(np.array(data_list), np.array(golden_data_list), rtol=1.e-4, atol=1.e-3)
    assert np.allclose(np.array(idx_list), np.array(golden_idx_list), rtol=1.e-5, atol=1.e-6) or \
            np.allclose(np.array(idx_list), np.array([1268, 1259, 1265, 1379, 1262, 1382, 1256, 1376, 1388, 1385, 1148, 1271, 1373, 1499, 1253, 1145, 1142, 1496, 1391, 1493, 1355, 1274, 1502, 1139, 1475, 1136, 1235, 1370, 1490, 1250, 1358, 1478, 810, 807, 808, 809, 813, 1715, 1595, 1718, 1598]), rtol=1.e-5, atol=1.e-6)
    assert np.allclose(golden_2, out_2, rtol=1.e-5, atol=1.e-6)
    print("==================================== PASS =========================================")

def get_temp_file_name():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".ser_data") as f:
        return  f.name

def test_custom_combine_non_max_suppression_post_sd():
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_combine_non_max_suppression_post(f, False)  # serialize
    with open(file_name, "rb") as f:
        test_custom_combine_non_max_suppression_post(f, True)  # deserialize

def test_custom_csum_plugin_sd():
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_csum_plugin(f,False) #serialize
    with open(file_name, "rb") as f:
        test_custom_csum_plugin(f,True) #deserialize

def test_custom_filter_sort_416_416_sd():
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_filter_sort_416_416(f, False)  # serialize
    with open(file_name, "rb") as f:
        test_custom_filter_sort_416_416(f, True)  # deserialize

def test_custom_filter_sort_pure_416_416_sd():
    import time
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_filter_sort_pure_416_416(f, False)  # serialize
        time.sleep(1)
    with open(file_name, "rb") as f:
        test_custom_filter_sort_pure_416_416(f, True)  # deserialize
        time.sleep(1)

def test_custom_filter_sort_720p_sd():
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_filter_sort_720p(f, False)  # serialize
    with open(file_name, "rb") as f:
        test_custom_filter_sort_720p(f, True)  # deserialize

def test_custom_filter_sort_pure_720p_sd():
    import time
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_filter_sort_pure_720p(f, False)  # serialize
        time.sleep(1)
    with open(file_name, "rb") as f:
        test_custom_filter_sort_pure_720p(f, True)  # deserialize
        time.sleep(1)

def test_box(serialize_file=None,deserialize=False):
    placeholder_0 = relay.var("input0", shape=(1, 13, 13, 12), dtype="float32")
    placeholder_1 = relay.var("input1", shape=(3, 2), dtype="float32")
    placeholder_2 = relay.var("input2", shape=(4,), dtype="int32")
    placeholder_3 = relay.var("input3", shape=(1, 2), dtype="int32")

    func = tvm.get_global_func("dl.relay.op._make.custom_boxes_plugin")
    boxes = func(placeholder_0, placeholder_1, placeholder_2,placeholder_3,"NHWC",1)

    net = relay.Function([placeholder_0,placeholder_1,placeholder_2,placeholder_3], boxes)
    mod = tvm.IRModule()
    mod["main"] = net
    mod = dl.relay.ir_pass.optimize(mod)
    params = {
            "input0":np.load((BasePath+"/box_golden/box_nv_input0.npy")),
            "input1":np.load((BasePath+"/box_golden/box_nv_input1.npy")),
            "input2":np.load((BasePath+"/box_golden/box_nv_input2.npy")),
            "input3":np.load((BasePath+"/box_golden/box_nv_input3.npy"))
    }
    
    return evaluate_with_nne(mod, params,"0",serialize_file=serialize_file,deseralize=deserialize)

def test_box_all():
    
    box_serialize_data = get_temp_file_name()

    #test1: serialize
    with open(box_serialize_data, "wb") as f:
        data = test_box(serialize_file=f, deserialize=False)
        golden = np.load((BasePath+"/box_golden/box_nv_output.npy"))
        print("max diff : ",np.abs(data - golden).max())
        assert np.abs(data - golden).max() < 0.0003

    #test2: deserialize
    with open(box_serialize_data, "rb") as f:
        data = test_box(serialize_file=f, deserialize=True)
        golden = np.load((BasePath+"/box_golden/box_nv_output.npy"))
        print("max diff : ",np.abs(data - golden).max())
        assert np.abs(data - golden).max() < 0.0003
    

def test_nms(serialize_file=None,deserialize=False):
    placeholder_0 = relay.var("input0", shape=(1, 80, 10647, 4), dtype="float32")
    placeholder_1 = relay.var("input1", shape=(1, 80, 16384), dtype="float32")
    placeholder_2 = relay.var("input2", shape=(1,), dtype="int32")
    placeholder_3 = relay.var("input3", shape=(1,), dtype="float32")
    placeholder_4 = relay.var("input4", shape=(1,), dtype="float32")
    placeholder_5 = relay.var("input5", shape=(1,80), dtype="int32")

    nms = dl.op.custom_non_max_suppression(placeholder_0, placeholder_1, placeholder_2,placeholder_3,placeholder_4,placeholder_5,0, "int32", 1)

    net = relay.Function([placeholder_0,placeholder_1,placeholder_2,placeholder_3,placeholder_4,placeholder_5], nms)
    mod = tvm.IRModule()
    mod["main"] = net
    mod = dl.relay.ir_pass.optimize(mod)
    # print(mod)
    # exit()
    params = {
            "input0":np.load((BasePath+"/nms_golden/nms_nv_input0.npy")),
            "input1":np.load((BasePath+"/nms_golden/nms_nv_input1.npy")),
            "input2":np.array([np.load((BasePath+"/nms_golden/nms_nv_input2.npy"))]).astype(np.int32),
            "input3":np.array([np.load((BasePath+"/nms_golden/nms_nv_input3.npy"))]).astype(np.float32),
            "input4":np.array([np.load((BasePath+"/nms_golden/nms_nv_input4.npy"))]).astype(np.float32),
            "input5":np.load((BasePath+"/nms_golden/nms_nv_input5.npy")),
    }
   
    return evaluate_with_nne(mod, params,"0",serialize_file=serialize_file,deseralize=deserialize)

def nms_compare_with_golden(data,golden0,golden1,golden2):
    valid_index = []
    for i in range(1):
        for j in range(80):
            for k in range(10647):
                    if(golden0[i,j,k] != 0):
                        valid_index.append([i,j,k])
    print(len(valid_index))
   
    for index in valid_index:
        print("diff: ",np.abs(golden0[index[0],index[1],index[2],index[3]]-data[0][index[0],index[1],index[2],index[3]]).max())
        assert(np.abs(golden0[index[0],index[1],index[2],index[3]]-data[0][index[0],index[1],index[2],index[3]]).max() == 0)

    #2
    valid_index = []
    for i in range(1):
        for j in range(80):
            for k in range(10647):
                    if(golden2[i,j,k] != 0):
                        valid_index.append([i,j,k])
    print(len(valid_index))
   
    for index in valid_index:
        print("diff: ",np.abs(golden2[index[0],index[1],index[2]].astype(np.uint8)-data[2][index[0],index[1],index[2]].astype(np.uint8)).max()," ",golden2[index[0],index[1],index[2]].astype(np.uint8)," ",data[2][index[0],index[1],index[2]].astype(np.uint8))
        assert(np.abs(golden2[index[0],index[1],index[2]].astype(np.uint8) - data[2][index[0],index[1],index[2]].astype(np.uint8)).max() == 0)
    
    valid_index = []
    for i in range(1):
        for j in range(80):
                    if(golden1[i,j] != 0):
                        valid_index.append([i,j])
    print(len(valid_index))
   
    for index in valid_index:
        print("diff: ",np.abs(golden1[index[0],index[1]] - data[1][index[0],index[1]]).max(), " ", golden1[index[0],index[1]], " ", data[1][index[0],index[1]])
        assert(np.abs(golden1[index[0],index[1]] - data[1][index[0],index[1]]).max() == 0)

def test_nms_all():
    golden0 = np.load((BasePath+"/nms_golden/nms_nv_output0.npy"))
    golden1 = np.load((BasePath+"/nms_golden/nms_nv_output1.npy"))
    golden2 = np.load((BasePath+"/nms_golden/nms_nv_output2.npy"))

    nms_serialize_data = get_temp_file_name()
    #1 test serialize
    with open(nms_serialize_data, "wb") as f:
        data = test_nms(serialize_file=f, deserialize=False) 
        nms_compare_with_golden(data,golden0,golden1,golden2)

    #2 test deserialize
    with open(nms_serialize_data, "rb") as f:
        data = test_nms(serialize_file=f, deserialize=True) 
        nms_compare_with_golden(data,golden0,golden1,golden2)
    

def test_nms_gather_box(serialize_file=None,deserialize=False):
    placeholder_0 = relay.var("input0", shape=(1, 1, 10647, 4), dtype="float32")
    placeholder_1 = relay.var("input1", shape=(1, 80, 16384), dtype="int32")
    placeholder_2 = relay.var("input2", shape=(1, 80), dtype="int32")
    boxes_gather = dl.op.custome_non_max_suppression_gather_boxes(placeholder_0, placeholder_1, placeholder_2)

    net = relay.Function([placeholder_0,placeholder_1,placeholder_2], boxes_gather)
    mod = tvm.IRModule()
    mod["main"] = net
    mod = dl.relay.ir_pass.optimize(mod)
    params = {
            "input0":np.load((BasePath+"/nms_gather_box_golden/nms_gather_box_nv_input0.npy")),
            "input1":np.load((BasePath+"/nms_gather_box_golden/nms_gather_box_nv_input1.npy")),
            "input2":np.load((BasePath+"/nms_gather_box_golden/nms_gather_box_nv_input2.npy"))
    }
    
    return evaluate_with_nne(mod, params,"0",serialize_file=serialize_file,deseralize=deserialize)

def gb_compare_with_golden(data,golden):
    valid_index = []
    for i in range(1):
        for j in range(80):
            for k in range(10647):
                for l in range(4):
                    if(golden[i,j,k,l] != 0.0):
                        valid_index.append([i,j,k,l])
    print(len(valid_index))
   
    for index in valid_index:
        print("diff: ",np.abs(golden[index[0],index[1],index[2],index[3]]-data[index[0],index[1],index[2],index[3]]).max())
        assert(np.abs(golden[index[0],index[1],index[2],index[3]]-data[index[0],index[1],index[2],index[3]]).max() == 0)

def test_nms_gather_box_all():
    golden = np.load((BasePath+"/nms_gather_box_golden/nms_gather_box_nv_output.npy"))

    nms_serialize_data = get_temp_file_name()
    #1 test serialize
    with open(nms_serialize_data, "wb") as f:
        data = test_nms_gather_box(serialize_file=f,deserialize=False)
        gb_compare_with_golden(data,golden)

    #2 test deserialize
    with open(nms_serialize_data, "rb") as f:
        data = test_nms_gather_box(serialize_file=f,deserialize=True)
        gb_compare_with_golden(data,golden)

def run_relay_graph(
    graph_def,
    input_data,
    input_node,
    num_output=1,
    target="cuda",
    outputs=None,
    layout="NHWC",
    ignore_in_shape=False,
    serialize_file=None,
    deserialize=False
):
    """
        Generic function to compile on relay and execute on tvm
    :param graph_def:
    :param input_data:
    :param input_node:
    :param num_output:
    :param target:
    :param outputs:
    :param layout:
    :return:
        For single output, it's a numpy array.
        For multiple outputs, they're a list of numpy array.

    """
    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, e in enumerate(input_node):
            if ignore_in_shape:
                shape_dict = None
            else:
                if input_data[i] is not None:
                    shape_dict[e] = input_data[i].shape
            if input_data[i] is not None:
                dtype_dict[e] = input_data[i].dtype
    else:
        if ignore_in_shape:
            shape_dict = None
        else:
            if input_data is not None:
                shape_dict = {input_node: input_data.shape}
        if input_data is not None:
            dtype_dict = {input_node: input_data.dtype}

    mod, params = relay.frontend.from_tensorflow(
        graph_def, layout=layout, shape=shape_dict, outputs=outputs
    )

    mod = dl.relay.ir_pass.optimize(mod)
    print("mod: \n",mod)
    
    if isinstance(input_data, list):
        for i, e in enumerate(input_node):
            if e:
                params[e] = input_data[i].astype(input_data[i].dtype)
    elif input_data is not None and input_node:
        params[input_node] = input_data.astype(input_data.dtype)

    return evaluate_with_nne(mod, params,"0",serialize_file=serialize_file,deseralize=deserialize)

def DlNMSCompareResult(tvm_output_all):
    for i in range(4):
        print(tvm_output_all[i].shape, "  ", tvm_output_all[i].dtype)

    for i in range(4):
        golden = np.load((BasePath+"/complete_nms_golden/golden_{}.npy".format(i)))
        print(np.abs(tvm_output_all[i][0:5] - golden[0:5]).max())
        if i == 1:
            assert np.abs(tvm_output_all[i][0:5] - golden[0:5]).max() < 2e-4
        elif i == 2:
            assert np.abs(tvm_output_all[i][0:5] - golden[0:5]).max() < 1e-3
        elif i == 3:
            assert tvm_output_all[i][0] == 5
        else:
            assert np.abs(tvm_output_all[i][0:5] - golden[0:5]).max() == 0.0

def compute_IOU(box1, box2):
    class BOX:
        def __init__(self,y0,x0,y1,x1):
            self.x0=x0
            self.y0=y0
            self.x1=x1
            self.y1=y1
    a = BOX(box1[0],box1[1],box1[2],box1[3])
    b = BOX(box2[0],box2[1],box2[2],box2[3])
    ymin_i = min(a.y0,a.y1);
    xmin_i = min(a.x0,a.x1);
    ymax_i = max(a.y0,a.y1);
    xmax_i = max(a.x0,a.x1);

    ymin_j = min(b.y0,b.y1);
    xmin_j = min(b.x0,b.x1);
    ymax_j = max(b.y0,b.y1);
    xmax_j = max(b.x0,b.x1);

    area_i = (ymax_i - ymin_i)*(xmax_i - xmin_i);
    area_j = (ymax_j - ymin_j)*(xmax_j - xmin_j);

    if(area_i <=0 or area_j <= 0):
        return 0.0

    interection_ymin = max(ymin_i,ymin_j);
    interection_xmin = max(xmin_i,xmin_j);
    interection_ymax = min(ymax_i,ymax_j);
    interection_xmax = min(xmax_i,xmax_j);

    interection_area = max((interection_ymax-interection_ymin),0)*max((interection_xmax-interection_xmin),0);

    return interection_area/(area_i+area_j-interection_area);

def test_iou():
    box1 = (140.7836,226.90552,390,651)
    box2 = (104,73,423,570)
    box3 = (186,27,394,309)
    print(compute_IOU(box1,box2))
    print(compute_IOU(box1,box3))
    print(compute_IOU(box2,box3))

def check_iou(boxes,iou_threshold):    
    box_num = len(boxes)

    start_index = 0
    while start_index < (box_num  - 1):
        box_1 = boxes[start_index]
        box_1 = np.reshape(box_1,(4,)).tolist()
        for i in range((start_index+1),box_num):
            box_2 = boxes[i]
            box_2 = np.reshape(box_2,(4,)).tolist()

            curren_iou = compute_IOU(box_1,box_2)
            # print("curren_iou: ",curren_iou," ",box_1,box_2)
            assert (curren_iou < iou_threshold), "iou greater than the threshold !"
        start_index += 1

def test_DlNonMaxSuppression():
    import time
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf

    with tf.Graph().as_default() as graph:
        box_concat_p = tf.placeholder(shape=(1, 1, 57960, 4), dtype=tf.float32, name="box_concat_p")
        box_scores_concat_p = tf.placeholder(
            shape=((1, 80,57960,)), dtype=tf.float32, name="box_scores_concat_p"
        )
        NMS_Const = tf.constant(20, dtype=tf.int32, shape=(), name="NMS_Const")
        iou_threshold = tf.constant(0.5, dtype=tf.float32, shape=(), name="iou_threshold")
        score_threshold = tf.constant(
            0.30000001192092896, dtype=tf.float32, shape=(), name="score_threshold"
        )

        graph_def = graph.as_graph_def()

        from tensorflow.core.framework import node_def_pb2,attr_value_pb2

        CombinedNonMaxSuppression_node_def = node_def_pb2.NodeDef()
        CombinedNonMaxSuppression_node_def.op = "DlNonMaxSuppression"
        CombinedNonMaxSuppression_node_def.name = (
            "NMS/combined_non_max_suppression/CombinedNonMaxSuppression"
        )
        CombinedNonMaxSuppression_node_def.attr["pad_per_class"].CopyFrom(
            attr_value_pb2.AttrValue(b=False)
        )
        CombinedNonMaxSuppression_node_def.attr["clip_boxes"].CopyFrom(
            attr_value_pb2.AttrValue(b=False)
        )
        CombinedNonMaxSuppression_node_def.input.append("box_concat_p")
        CombinedNonMaxSuppression_node_def.input.append("box_scores_concat_p")
        CombinedNonMaxSuppression_node_def.input.append("NMS_Const")
        CombinedNonMaxSuppression_node_def.input.append("NMS_Const")
        CombinedNonMaxSuppression_node_def.input.append("iou_threshold")
        CombinedNonMaxSuppression_node_def.input.append("score_threshold")

        graph_def.node.extend([CombinedNonMaxSuppression_node_def])

        # out_graph_def_file = "/LocalRun/wang.tang/Project/tvm/tvm_notes/graph_def_tools/nhwc_to_nchw/CombinedNonMaxSuppression.pb"
        # with tf.gfile.GFile(out_graph_def_file, 'wb') as f:
        #     f.write(graph_def.SerializeToString())

        box_concat_np = np.load((BasePath+"/complete_nms_golden/nms_input_box.npy"))
        box_scores_concat_np = np.load((BasePath+"/complete_nms_golden/nms_input_score.npy"))
        
        nms_serialize_data = get_temp_file_name()
        pdb.set_trace()
        #1 test serialize
        with open(nms_serialize_data, "wb") as f:
            tvm_output_all = run_relay_graph(
                graph_def,
                [box_concat_np, box_scores_concat_np],
                ["box_concat_p", "box_scores_concat_p"],
                target="cuda",
                num_output=1,
                outputs=["NMS/combined_non_max_suppression/CombinedNonMaxSuppression:0",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:1",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:2",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:3"],
                layout="NCHW",
                serialize_file=f,deserialize=False)

            DlNMSCompareResult(tvm_output_all)
            print("==================================== PASS =========================================")
            time.sleep(1)

        #2 test deserialize
        with open(nms_serialize_data, "rb") as f:
            tvm_output_all = run_relay_graph(
                graph_def,
                [box_concat_np, box_scores_concat_np],
                ["box_concat_p", "box_scores_concat_p"],
                target="cuda",
                num_output=1,
                outputs=["NMS/combined_non_max_suppression/CombinedNonMaxSuppression:0",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:1",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:2",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:3"],
                layout="NCHW",
                serialize_file=f,deserialize=True)

            DlNMSCompareResult(tvm_output_all)
            print("==================================== PASS =========================================")
            time.sleep(1)

def test_DlNonMaxSuppression_one_class_IOU():
    import time
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf

    box_shape = (1, 1, 1200, 4)
    score_shape = (1, 1,1200)
    with tf.Graph().as_default() as graph:
        box_concat_p = tf.placeholder(shape=box_shape, dtype=tf.float32, name="box_concat_p")
        box_scores_concat_p = tf.placeholder(
            shape=score_shape, dtype=tf.float32, name="box_scores_concat_p"
        )
        NMS_Const = tf.constant(20, dtype=tf.int32, shape=(), name="NMS_Const")
        iou_threshold = tf.constant(0.5, dtype=tf.float32, shape=(), name="iou_threshold")
        score_threshold = tf.constant(
            0.30000001192092896, dtype=tf.float32, shape=(), name="score_threshold"
        )

        graph_def = graph.as_graph_def()

        from tensorflow.core.framework import node_def_pb2,attr_value_pb2

        CombinedNonMaxSuppression_node_def = node_def_pb2.NodeDef()
        CombinedNonMaxSuppression_node_def.op = "DlNonMaxSuppression"
        CombinedNonMaxSuppression_node_def.name = (
            "NMS/combined_non_max_suppression/CombinedNonMaxSuppression"
        )
        CombinedNonMaxSuppression_node_def.attr["pad_per_class"].CopyFrom(
            attr_value_pb2.AttrValue(b=False)
        )
        CombinedNonMaxSuppression_node_def.attr["clip_boxes"].CopyFrom(
            attr_value_pb2.AttrValue(b=False)
        )
        CombinedNonMaxSuppression_node_def.input.append("box_concat_p")
        CombinedNonMaxSuppression_node_def.input.append("box_scores_concat_p")
        CombinedNonMaxSuppression_node_def.input.append("NMS_Const")
        CombinedNonMaxSuppression_node_def.input.append("NMS_Const")
        CombinedNonMaxSuppression_node_def.input.append("iou_threshold")
        CombinedNonMaxSuppression_node_def.input.append("score_threshold")

        graph_def.node.extend([CombinedNonMaxSuppression_node_def])

        box_concat_np = np.load((BasePath+"/complete_nms_golden/nms_input_box.npy"))
        box_scores_concat_np = np.load((BasePath+"/complete_nms_golden/nms_input_score.npy"))

        base_score = 0.8
        base_box = [50.0, 150.0, 150.0, 50.0]
        box_concat_np = []
        box_scores_concat_np = []
        for i in range(score_shape[2]):
            box_scores_concat_np.append((base_score + np.random.uniform(-0.2,0.2,()).tolist()))
            box_concat_np.append([(base_box[0] + np.random.uniform(-50,50,()).tolist()),
                                  (base_box[1] + np.random.uniform(-50,50,()).tolist()),
                                  (base_box[2] + np.random.uniform(-50,50,()).tolist()),
                                  (base_box[3] + np.random.uniform(-50,50,()).tolist())])
        box_scores_concat_np = np.reshape(box_scores_concat_np,score_shape).astype("float32")
        box_concat_np = np.reshape(box_concat_np,box_shape).astype("float32")

        nms_serialize_data = get_temp_file_name()
        pdb.set_trace()
        #1 test serialize
        with open(nms_serialize_data, "wb") as f:
            tvm_output_all = run_relay_graph(
                graph_def,
                [box_concat_np, box_scores_concat_np],
                ["box_concat_p", "box_scores_concat_p"],
                target="cuda",
                num_output=1,
                outputs=["NMS/combined_non_max_suppression/CombinedNonMaxSuppression:0",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:1",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:2",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:3"],
                layout="NCHW",
                serialize_file=f,deserialize=False)

            valid_num = tvm_output_all[3].tolist()[0]
            valid_box = tvm_output_all[1][:valid_num]
            check_iou(valid_box,0.5)
            print("==================================== PASS =========================================")
            time.sleep(1)

        #2 test deserialize
        with open(nms_serialize_data, "rb") as f:
            tvm_output_all = run_relay_graph(
                graph_def,
                [box_concat_np, box_scores_concat_np],
                ["box_concat_p", "box_scores_concat_p"],
                target="cuda",
                num_output=1,
                outputs=["NMS/combined_non_max_suppression/CombinedNonMaxSuppression:0",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:1",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:2",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:3"],
                layout="NCHW",
                serialize_file=f,deserialize=True)

            valid_num = tvm_output_all[3].tolist()[0]
            valid_box = tvm_output_all[1][:valid_num]
            check_iou(valid_box,0.5)
            print("==================================== PASS =========================================")
            time.sleep(1)
            
def test_DlNonMaxSuppression_one_class_IOU_custom():
    import time
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf

    box_shape = (1, 1, 10, 4)
    score_shape = (1, 1,10)
    with tf.Graph().as_default() as graph:
        box_concat_p = tf.placeholder(shape=box_shape, dtype=tf.float32, name="box_concat_p")
        box_scores_concat_p = tf.placeholder(
            shape=score_shape, dtype=tf.float32, name="box_scores_concat_p"
        )
        NMS_Const = tf.constant(20, dtype=tf.int32, shape=(), name="NMS_Const")
        iou_threshold = tf.constant(0.5, dtype=tf.float32, shape=(), name="iou_threshold")
        score_threshold = tf.constant(
            0.30000001192092896, dtype=tf.float32, shape=(), name="score_threshold"
        )

        graph_def = graph.as_graph_def()

        from tensorflow.core.framework import node_def_pb2,attr_value_pb2

        CombinedNonMaxSuppression_node_def = node_def_pb2.NodeDef()
        CombinedNonMaxSuppression_node_def.op = "DlNonMaxSuppression"
        CombinedNonMaxSuppression_node_def.name = (
            "NMS/combined_non_max_suppression/CombinedNonMaxSuppression"
        )
        CombinedNonMaxSuppression_node_def.attr["pad_per_class"].CopyFrom(
            attr_value_pb2.AttrValue(b=False)
        )
        CombinedNonMaxSuppression_node_def.attr["clip_boxes"].CopyFrom(
            attr_value_pb2.AttrValue(b=False)
        )
        CombinedNonMaxSuppression_node_def.input.append("box_concat_p")
        CombinedNonMaxSuppression_node_def.input.append("box_scores_concat_p")
        CombinedNonMaxSuppression_node_def.input.append("NMS_Const")
        CombinedNonMaxSuppression_node_def.input.append("NMS_Const")
        CombinedNonMaxSuppression_node_def.input.append("iou_threshold")
        CombinedNonMaxSuppression_node_def.input.append("score_threshold")

        graph_def.node.extend([CombinedNonMaxSuppression_node_def])

        box_concat_np = np.load((BasePath+"/complete_nms_golden/nms_input_box.npy"))
        box_scores_concat_np = np.load((BasePath+"/complete_nms_golden/nms_input_score.npy"))

        base_score = 0.8
        box_concat_np = [[3.07e2,2.77e2,3.16e2,2.85e2],
                         [3.09e2,2.60e2,3.17e2,2.71e2],
                         [3.20e2,2.75e2,3.36e2,2.85e2],
                         [3.16e2,2.55e2,3.44e2,2.74e2],
                         [2.92e2,2.75e2,3.00e2,2.84e2],
                         [2.91e2,2.59e2,3.00e2,2.67e2],
                         [-7.67e-37,-1.25e-35,-5.05e-35,-1.28e-38],
                         [-1.94e-37,-1.22e-38,-3.09e-33,-1.81e-35],
                         [-1.19e-38,-3.06e-36,-3.13e-36,-1.28e-38],
                         [-3.13e-36,-1.21e-38,-4.83e-35,-1.81e-35]]
        box_scores_concat_np = [9.54e-1,9.25e-1,8.17e-1,6.92e-1,4.47e-1,3.73e-1,1.17e-38,-1.33e-38,-1.20e-38,-1.18e-38]
            
        box_scores_concat_np = np.reshape(box_scores_concat_np,score_shape).astype("float32")
        box_concat_np = np.reshape(box_concat_np,box_shape).astype("float32")

        nms_serialize_data = get_temp_file_name()
        #1 test serialize
        with open(nms_serialize_data, "wb") as f:
            tvm_output_all = run_relay_graph(
                graph_def,
                [box_concat_np, box_scores_concat_np],
                ["box_concat_p", "box_scores_concat_p"],
                target="cuda",
                num_output=1,
                outputs=["NMS/combined_non_max_suppression/CombinedNonMaxSuppression:0",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:1",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:2",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:3"],
                layout="NCHW",
                serialize_file=f,deserialize=False)

            valid_num = tvm_output_all[3].tolist()[0]
            valid_box = tvm_output_all[1][:valid_num]
            check_iou(valid_box,0.5)
            print("==================================== PASS =========================================")
            time.sleep(1)

        #2 test deserialize
        with open(nms_serialize_data, "rb") as f:
            tvm_output_all = run_relay_graph(
                graph_def,
                [box_concat_np, box_scores_concat_np],
                ["box_concat_p", "box_scores_concat_p"],
                target="cuda",
                num_output=1,
                outputs=["NMS/combined_non_max_suppression/CombinedNonMaxSuppression:0",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:1",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:2",
                        "NMS/combined_non_max_suppression/CombinedNonMaxSuppression:3"],
                layout="NCHW",
                serialize_file=f,deserialize=True)

            valid_num = tvm_output_all[3].tolist()[0]
            valid_box = tvm_output_all[1][:valid_num]
            check_iou(valid_box,0.5)
            print("==================================== PASS =========================================")
            time.sleep(1)


def test_hk_nms_all_plugin():
    print("---test_hk_nms_all_plugin---")

    # test_arange()
    const226 = 20

    # need specify
    const227 = relay.ones((5,2), dtype="float32")
    const228 = 5
    const228 = 40

    # im_info_var = relay.var("im_info",(1,3), dtype="int32") 
    # data153_var = relay.var("data153", (1, 65, 20, 32), dtype="float16")

    # need specify
    im_info_var = relay.ones((1,3),dtype="float32")
    data153_var = relay.ones((1, 65, 20, 32),dtype="float32")

    # need specify
    data154 = relay.take(data153_var, indices=relay.arange(relay.const(0),relay.const(20),dtype="int32"), axis=1, batch_dims=0, mode="clip")
    data155 = relay.transpose(data154, (0, 2, 3, 1))
    im_info156 = relay.divide(im_info_var, relay.const(16, dtype="float32"))
    data157 = relay.round(im_info156)

    
    # % 158 = cast( % 155, dtype = "float32") / *ty = Tensor[(1, 20, 32, 20), float32] * /;
    # % 159 = cast( % 157, dtype = "int32") / *ty = Tensor[(1, 3), int32] * /;
    # % 160 = cast( % im_info, dtype = "int32") / *ty = Tensor[(1, 3), int32] * /;
    data158 = relay.cast(data155, "float32")
    data159 = relay.cast(data157, "int32")
    data160 = relay.cast(im_info_var, "int32")

    data161_boxes = tvm.get_global_func("dl.relay.op._make.custom_boxes_plugin") (
    data158, const227, data159, data160, "NHWC", 0)
    
    data162 = relay.take(data153_var, indices=relay.arange(relay.const(0),relay.const(5),dtype="int32"), axis=1, mode="fast")
    data163 = relay.take(data153_var, indices=relay.arange(relay.const(25),relay.const(65),dtype="int32"), axis=1, mode="fast")
    # data163 = relay.take(data153_var, indices=relay.const(40, dtype="int32"), axis=1, mode="fast")
    data164 = relay.reshape(data163,newshape=[1, 5, 8, 20, 32]) #8x3200
    data165 = relay.transpose(data164, axes=[0, 2, 1, 3, 4])
    data166 = relay.nn.softmax(data165, axis=1)
    data167 = relay.sigmoid(data162) #1x5x20x32
    data168 = relay.reshape(data166,newshape=[1, 40, 20, 32]) # 8x3200
    data169 = _expr.Tuple((data167, data168))

    iou_threshold = 0.45 #attr['iouThreshold']
    score_threshold = 0.2 #attr['scoreThreshold']
    # num_classes = attr['numClasses']
    num_classes = 8
    # top_k = attr['topK']
    # keepTopK = attr['keepTopK']
    keepTopK = -1
    # background_label_id = attr['backgroundLabelId']
    score_threshold_scalar = float(score_threshold)

    import math
    log2block = int(math.ceil(math.log2(3200)))
    total_length = pow(2, log2block)

    # need specify
    pre_s = 8
    sorted_scores_idx = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        # data169,
        _expr.Tuple([data168]),
        score_threshold_scalar,
        False,
        "float32",
        "int32",
        [[pre_s]+[total_length],[pre_s]+[total_length]]
        )

    data176_sorted_scores, data171_sorted_idx, data173_sort_size = _expr.TupleWrapper(sorted_scores_idx, 3)
    dtype="float32"
    data172_sorted_idx = relay.reshape(data171_sorted_idx,newshape=[1, 8, -1]) #/* ty=Tensor[(1, 8, 4096), int32] */;
    data174_sort_size = relay.reshape( data173_sort_size, newshape = [1, 8]) #/ *ty = Tensor[(1, 8), int32] * /;
    data175_boxes = relay.reshape( data161_boxes, newshape = [1, 1, -1, 4])   # / *ty = Tensor[(1, 1, 3200, 4), float32] * /;


    data177_boxes_gather = tvm.get_global_func( "dl.relay.op._make.custome_non_max_suppression_gather_boxes")(data175_boxes, data172_sorted_idx, data174_sort_size)

    data178_sorted_scores = relay.reshape( data176_sorted_scores, newshape = [1, 8, -1]) #/ *ty = Tensor[(1, 8, 4096), float32] * /;

    iou_threshold_var = relay.const(np.array([iou_threshold]).astype(dtype), dtype=dtype)
    score_threshold_var = relay.const(np.array([score_threshold]).astype(dtype), dtype=dtype)
    max_output_size_per_class_var = relay.const(np.array([keepTopK]).astype("int32"), dtype="int32")

    # boxes_ids_count = tvm.get_global_func("dl.relay.op._make.custom_non_max_suppression")(boxes_gather,
    #         sorted_scores,
    #         max_output_size_per_class_var,
    #         iou_threshold_var,
    #         score_threshold_var,
    #         sort_size, 0, "int32", 1)

    data179_boxes_ids_count = dl.op.custom_non_max_suppression(data177_boxes_gather,
        data178_sorted_scores,
        max_output_size_per_class_var,
        iou_threshold_var,
        score_threshold_var,
        data174_sort_size, 0, "int32", 1)

    data181_selected_ids, data180_count, sort_size = _expr.TupleWrapper(data179_boxes_ids_count, 3)

    data182_csum_value = tvm.get_global_func("dl.relay.op._make.custom_csum")(data180_count,-1,1)

    data183 = tvm.get_global_func("dl.relay.op._make.custom_combine_non_max_suppression_post")(
        data177_boxes_gather,
        data178_sorted_scores,
        data181_selected_ids,
        data180_count, data182_csum_value,
        9999)

    data184, data187, data185 = _expr.TupleWrapper(data183, 3)
    data186 = relay.reshape(data185, newshape=[-1, 1]) #/* ty=Tensor[(9999, 1), float32] */;

    data188 = relay.cast( data184, dtype="float32") #/ *ty = Tensor[(9999, 2), float32] * /;
    data189 = relay.cast(data186, dtype="float32") #/ *ty = Tensor[(9999, 1), float32] * /;
    data190 = relay.cast(data187, dtype="float32") #/ *ty = Tensor[(9999, 4), float32] * /;
    data191 = _expr.Tuple((data188, data189, data190))
    data192 = relay.cast(data172_sorted_idx, dtype="float32") #/ *ty = Tensor[(1, 8, 4096), float32] * /;
    data193 = relay.cast(data174_sort_size, dtype="float32") #/ *ty = Tensor[(1, 8), float32] * /;
    data194 = relay.concatenate(data191, axis=1)  #/ *ty = Tensor[(9999, 7), float32] * /;
    data195 = _expr.Tuple((data175_boxes, data192, data193, data194))
    # output = relay.concatenate(data195, axis=1)
    

    im_info_data = np.zeros((1,3),dtype = np.int32)
    data153_data = np.zeros((1,65,20,32),dtype = np.float16)

    params = {
        "im_info": im_info_data,
        "data153": im_info_data,
    }
    
    output = data195
    net = relay.Function([], output)
    mod = tvm.IRModule()
    mod["main"] = net
    print(mod)
    mod = dl.relay.ir_pass.optimize(mod)



    # outmod = tvm.IRModule.from_expr(output)
    # print(outmod)
    # print("hello world")

    # # mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)
    json_mod = tvm.ir.save_json(mod)
    file_name = "hktest.rlym"
    with open(file_name, "w+") as f:
        f.write(json_mod)

    mod = dl.relay.ir_pass.optimize(mod)
    output = evaluate_with_nne(mod, params,"0", serialize_file=None, deseralize=False)
    print("end")
    # print(output.shape)
    # result = run_mod(mod, params.copy(), "nne", open_opt=True,
    #             serialize_file=None, deseralize=False)



def test_relay_func():
    x = relay.var("x", shape=(1, 1000), dtype="float32")
    x1 = relay.var("x1", shape=(1, 1000), dtype="float32")
    y = relay.add(x, x)
    out = relay.subtract(y, x1)
    func = relay.Function([x,x1], out)
    mod = tvm.IRModule.from_expr(func)
    print(mod)
    print("hello")


if __name__ == '__main__':
    # test_custom_filter_sort_pure_416_416_sd()
    # test_custom_filter_sort_pure_720p_sd()                     #tvm.get_global_func
    # test_DlNonMaxSuppression()                                 # run_relay_graph
    # test_DlNonMaxSuppression_one_class_IOU()
    # test_DlNonMaxSuppression_one_class_IOU_custom()
    # test_custom_filter_sort_720p_sd()
    # test_custom_filter_sort_416_416_sd()
    # test_custom_combine_non_max_suppression_post_sd()
    # test_custom_csum_plugin_sd()
    # test_box_all()                                            # dl.op.custome_non_max_suppression_gather_boxes
    # test_nms_all()                                          # dl.op.custom_non_max_suppression
    # test_nms_gather_box_all()
    # print("yolov3_plugin pass")

    ## hk nms bug test
    # test_relay_func()
    test_hk_nms_all_plugin()
from __future__ import absolute_import, print_function
import tvm
from tvm import relay
import numpy as np
import dl

from tvm.topi.utils import get_const_tuple

def to_numpy(out):
    if isinstance(out, tvm.nd.NDArray):
        # Single result
        return out.asnumpy()
    else:
        # Multiple results
        return [r.asnumpy() for r in out]

import sys
import os

front_end = os.path.join(os.path.dirname(__file__),"../front_end.py")
import importlib.util
spec = importlib.util.spec_from_file_location("plugin_tvm", front_end)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
from tvm.relay import op as _op, expr as _expr

dlnne_so=os.path.join(os.path.dirname(__file__),"../dlnne_plugin_build/dlnne_plugin.so")
BasePath=os.path.dirname(os.path.abspath(__file__))


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
            parser.register_user_op(dlnne_so,front_end,"custom_op")

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

            # builder.max_batch_size = 1
            context = engine.create_execution_context(cluster_cfg)
            nb = engine.num_bindings
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


def test_custom_filter_sort_single_input_416_416(serialize_file=None,deseralize=False):
    tu77 = relay.var("tu77", shape=[1, 3, 13, 13], dtype="float16")
    tu77_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu77.npy").astype(np.float16), (1, 3, 13, 13))
    tu78 = relay.var("tu78", shape=[1, 240, 13, 13], dtype="float16")
    tu78_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu78.npy").astype(np.float16), (1, 240, 13, 13))

    params = {
        "tu77": tu77_np,
        "tu78": tu78_np,
    }

    custom_filter_sort_single_input = tvm.get_global_func("dl.relay.op._make.custom_filter_sort_single_input")(
        _expr.Tuple([tu77, tu78]),
        0.3,
        False,
        "float32",
        "int32",
        [[80]+[1024],[80]+[1024]],
        )

    func = relay.Function([tu77, tu78], custom_filter_sort_single_input);
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
    # assert np.allclose(np.array(idx_list), np.array(golden_idx_list), rtol=1.e-5, atol=1.e-6)
    assert np.allclose(golden_2, out_2, rtol=1.e-5, atol=1.e-6)
    print("==================================== PASS =========================================")

def test_custom_filter_sort_single_input_720p(serialize_file=None,deseralize=False):
    tu94 = relay.var("tu94", shape=[1, 3, 23, 40], dtype="float16")
    tu94_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu94.npy").astype(np.float16), (1, 3, 23, 40))
    tu95 = relay.var("tu95", shape=[1, 240, 23, 40], dtype="float16")
    tu95_np = np.reshape(np.load(BasePath+"/filter_sort_golden/tu95.npy").astype(np.float16), (1, 240, 23, 40))

    params = {
        "tu94": tu94_np,
        "tu95": tu95_np,
    }

    custom_filter_sort_single_input = tvm.get_global_func("dl.relay.op._make.custom_filter_sort_single_input")(
        _expr.Tuple([tu94, tu95]),
        0.3,
        False,
        "float32",
        "int32",
        [[80]+[1024],[80]+[1024]],
        )

    func = relay.Function([tu94, tu95], custom_filter_sort_single_input);
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
    # assert np.allclose(np.array(idx_list), np.array(golden_idx_list), rtol=1.e-5, atol=1.e-6)
    assert np.allclose(golden_2, out_2, rtol=1.e-5, atol=1.e-6)
    print("==================================== PASS =========================================")

def get_temp_file_name():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".ser_data") as f:
        return  f.name


def test_custom_filter_sort_single_input_416_416_sd():
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_filter_sort_single_input_416_416(f, False)  # serialize
    # with open(file_name, "rb") as f:
    #     test_custom_filter_sort_416_416(f, True)  # deserialize

def test_custom_filter_sort_single_input_720p_sd():
    file_name = get_temp_file_name()
    with open(file_name, "wb") as f:
        test_custom_filter_sort_single_input_720p(f, False)  # serialize
    # with open(file_name, "rb") as f:
    #     test_custom_filter_sort_720p(f, True)  # deserialize

 

if __name__ == '__main__':
    test_custom_filter_sort_single_input_720p_sd()
    test_custom_filter_sort_single_input_416_416_sd()

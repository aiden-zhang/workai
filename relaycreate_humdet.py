from __future__ import absolute_import, print_function
import tvm
from tvm import relay
import numpy as np
import dl
import pdb
from tvm.topi.utils import get_const_tuple
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay.frontend.common import infer_type as _infer_type
#from tvm.relay.frontend.onnx import (
#    OnnxOpConverter,
#    _get_convert_map,
#)
from tvm.target import override_native_generic_func
from tvm.relay.op.strategy.generic import wrap_topi_schedule
from tvm import te

#original_get_convert_map = _get_convert_map


def to_numpy(out):
    if isinstance(out, tvm.nd.NDArray):
        # Single result
        return out.asnumpy()
    else:
        # Multiple results
        return [r.asnumpy() for r in out]

import sys
import os

# KERNEL_PATH=os.path.dirname(os.path.abspath(__file__))
# KERNEL_PATH=os.path.join(KERNEL_PATH,"./dlnne_plugin/plugin/kernel")

# if ""==os.getenv("YOLOV3_PLUGIN_KERNEL_PATH",""):
#     os.environ["YOLOV3_PLUGIN_KERNEL_PATH"]=KERNEL_PATH

import dl
from dl import op
import pytest

# BasePath=os.path.dirname(os.path.abspath(__file__))
# plugin_tvm_so=os.path.join(BasePath,"./dlnne_plugin_build/libyolov3_opt_plugin.so")
# TVM_TVM_REGISTER_SO_NAME = os.path.join(BasePath,"./dlnne_plugin_build/libyolov3_opt_tvm.so")
# op.load_op_library(plugin_tvm_so)
# op.load_op_library(TVM_TVM_REGISTER_SO_NAME)


# front_end = os.path.join(BasePath,"./front_end.py")
# import importlib.util
# spec = importlib.util.spec_from_file_location("plugin_tvm", front_end)
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)
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
            # parser.register_user_op(TVM_TVM_REGISTER_SO_NAME,front_end,"custom_op")

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

            # builder.max_batch_size = 1
            context = engine.create_execution_context(cluster_cfg)

            batch_size = 1


            def _evaluate(inputs_dict):
                outputs = []
                keys = []
                bindings = []
                batch_input_shape = []
                # pdb.set_trace()
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


def relay_create_and_nne_infer():
    print("---test_hk_nms_all_plugin---")

#   %0 = cast(%data, dtype="float32") /* ty=Tensor[(1, 3, 512, 512), float32] */;
#   %1 = reshape(meta[relay.Constant][0] /* ty=Tensor[(1), float32] */, newshape=[1, 1, 1, 1]) /* ty=Tensor[(1, 1, 1, 1), float32] */;
#   %2 = subtract(%0, %1) /* ty=Tensor[(1, 3, 512, 512), float32] */;
#   %3 = multiply(%2, meta[relay.Constant][1] /* ty=Tensor[(1), float32] */) /* ty=Tensor[(1, 3, 512, 512), float32] */;
#   %4 = dl.quantize(%3, out_dtype="int8", output_scales=[0.03125f64], output_zero_points=[0]) /* ty=Tensor[(1, 3, 512, 512), int8] */;
#   %5 = (%4, meta[relay.Constant][2] /* ty=Tensor[(32, 3, 5, 5), int8] */, meta[relay.Constant][3] /* ty=Tensor[(32), int32] */, meta[relay.Constant][4] /* ty=Tensor[(32), float32] */, meta[relay.Constant][5] /* ty=Tensor[(32), int32] */, meta[relay.Constant][6] /* ty=Tensor[(32), int32] */, meta[relay.Constant][7] /* ty=Tensor[(32), int32] */, meta[relay.Constant][8] /* ty=Tensor[(32), int32] */);
#   %6 = dl.quantized_conv2d(%5, strides=[2, 2], channels=32, kernel_size=[5, 5], bias_layout="", out_layout="NCHW", out_dtype="float16", pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]], lhs_scales=[0.03125f64], lhs_zero_points=[0], rhs_scales=[0.000746846f, 0.00116217f, 0.000849605f, 0.000771075f, 0.000225186f, 0.000724852f, 0.000750333f, 0.000852883f, 0.000980526f, 0.000830263f, 0.000703037f, 0.000956059f, 0.00050357f, 0.000813961f, 0.000852734f, 0.000585765f, 0.000592858f, 0.000957042f, 0.000798076f, 0.000767916f, 0.000811219f, 0.000490785f, 0.000601888f, 0.000550091f, 0.000216305f, 0.000699013f, 0.000892669f, 0.000895321f, 0f, 0.000638604f, 0.000633508f, 0.0010047f], rhs_zero_points=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], scales=[0.0625f64], zero_points=[0]) /* ty=Tensor[(1, 32, 256, 256), float16] */;
#   %7 = nn.relu(%6) /* ty=Tensor[(1, 32, 256, 256), float16] */;
#   %8 = nn.max_pool2d(%7, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True) /* ty=Tensor[(1, 32, 128, 128), float16] */;
#   %9 = dl.quantize(%8, out_dtype="int8", output_scales=[0.0625f64], output_zero_points=[0]) /* ty=Tensor[(1, 32, 128, 128), int8] */;
  


    # input = relay.ones((1,96,100,100), dtype="float32")
    input = relay.var('data', shape=(1,3,512,512), dtype="uint8")
    d0 = relay.cast(input, dtype="float32")
    d1 = relay.reshape( _expr.const(np.ones(1,).astype('float32')),newshape=[1, 1, 1, 1])
    d2 = relay.subtract(d0,d1)
    d3 = relay.multiply( d2,  _expr.const(np.ones(1,).astype('float32')) )

    d4 = dl.relay.op.qnn.quantize(d3, output_scales=[0.015625], output_zero_points=[0],out_dtype="int8")

    d5 = [d4, _expr.const( np.ones((32,3,5,5)).astype('int8')), _expr.const(np.ones(32,).astype('int32'))]
    d6 = dl.relay.op.qnn.quantized_conv2d( 
        d5,
        strides=(2, 2),
        pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]], 
        channels=32,
        kernel_size=(5, 5),
        data_layout="NCHW",
        out_dtype="float16",
 
        data_scales=[0.015625], 
        data_zero_points=[0], 
        weight_scales=[0.015],
        weight_zero_points=[0], 
        scales=[0.0078125], 
        zero_points=[0]
        )
    d7 = relay.nn.relu(d6)
    d8 = relay.nn.max_pool2d(d7)
    d9 = dl.relay.op.qnn.quantize(d8, output_scales=[0.015625], output_zero_points=[0],out_dtype="int8")
    params = {
        "data": np.ones((1,3,512,512), dtype="float32"),
        # "im_info": im_info_data,
        # "data153": im_info_data,
    }
    
    net = d9
    func = relay.Function([input], net)
    mod = tvm.IRModule()
    mod["main"] = func
    print(mod)

    mod = tvm.relay.transform.InferType()(mod)
    json_mod = tvm.ir.save_json(mod)
    file_name = "humdet_head_9_nodes.rlym"
    with open(file_name, "w+") as f:
        f.write(json_mod)

    # mod = dl.relay.ir_pass.optimize(mod) 
    # pdb.set_trace()
    output = evaluate_with_nne(mod, params,"0", serialize_file=None, deseralize=False)
    for i in np.arange(len(output)):
        print(f"{output[i].shape}")


if __name__ == '__main__':
    relay_create_and_nne_infer()

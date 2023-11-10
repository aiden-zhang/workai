from __future__ import absolute_import, print_function
import tvm
from tvm import relay
import numpy as np
import dl
import pdb
from tvm.topi.utils import get_const_tuple
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay.frontend.common import infer_type as _infer_type
# from tvm.relay.frontend.onnx import (
#     OnnxOpConverter,
#     _get_convert_map,
# )
# from tvm.target import override_native_generic_func
# from tvm.relay.op.strategy.generic import wrap_topi_schedule
# from tvm import te

# original_get_convert_map = _get_convert_map


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
KERNEL_PATH=os.path.join(KERNEL_PATH,"./dlnne_plugin/plugin/kernel")

if ""==os.getenv("YOLOV3_PLUGIN_KERNEL_PATH",""):
    os.environ["YOLOV3_PLUGIN_KERNEL_PATH"]=KERNEL_PATH

import dl
from dl import op
import pytest

BasePath=os.path.dirname(os.path.abspath(__file__))
plugin_tvm_so=os.path.join(BasePath,"./dlnne_plugin_build/libyolov3_opt_plugin.so")
TVM_TVM_REGISTER_SO_NAME = os.path.join(BasePath,"./dlnne_plugin_build/libyolov3_opt_tvm.so")
op.load_op_library(plugin_tvm_so)
op.load_op_library(TVM_TVM_REGISTER_SO_NAME)


front_end = os.path.join(BasePath,"./front_end.py")
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
            builder.config.dump_dot=False
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


def test_hk_nms_all_plugin():
    print("---add new operators---")
    data_main_input = relay.ones((1,64,20,32), dtype="float32")
    
    data0 = dl.relay.op.qnn.quantize(data_main_input, output_scales=[0.125], output_zero_points=[0],out_dtype="int8")

    constant0=relay.ones((32,64,1,1),dtype="int8")
    # constant1 = relay.ones([32],dtype='int32')
    constant1=relay.const([32],dtype='int32')
    # constant2=relay.const(32,dtype='float64') 
    # constant3=relay.const(32,dtype='int32')
    # constant4=relay.const(32,dtype='int32')
    # constant5=relay.const(32,dtype='int32')
    # constant6=relay.const(32,dtype='int32')
    # data1 = _expr.Tuple((data0, constant0, constant1, constant2, constant3, constant4, constant5, constant6))
    # data1 = _expr.Tuple((data0, constant0))
    data2 = dl.relay.op.qnn.quantized_conv2d( [data0, _expr.const( np.ones((32,64,1,1)).astype('int8')), _expr.const(np.ones(32,).astype('int32'))], channels=32,kernel_size=(1, 1),data_layout="NCHW",out_dtype="float16",
        pad_width=((0, 0), (0, 0), (0, 0), (0, 0)), 
        data_scales=[0.125], 
        data_zero_points=[0], 
        weight_scales=[0.0031805, 0.00191939, 0.00179207, 0.00185061, 0.00406229, 0.00164783, 0.00242984, 
                        0.00334871, 0.00508463, 0.00436497, 0.00151861, 0.00233459, 0.00235581, 0.00226212, 
                        0.00232375, 0.00227082, 0.00371957, 0.00309527, 0.00321388, 0.00223494, 0.00207329,
                        0.00298762, 0.00248945, 0.00419426, 0.0054574, 0.00338709, 0.00308597, 0.00375617, 
                        0.00243366, 0.00190258, 0.0023638, 0.00210512],
        weight_zero_points=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        scales=[0.0625], 
        zero_points=[0]
        )

    data3 = relay.nn.relu(data2)
    data4 = dl.relay.op.qnn.quantize(data3, output_scales=[0.125], output_zero_points=[0],out_dtype="int8")

    # constant7=relay.zeros((32,32,1,1),dtype="int8")
    constant7=np.zeros((32,32,1,1)).astype('int8')
    constant8=np.ones(32,).astype('int32')
    # constant8=relay.const(32,dtype='int32') 
    # constant9=relay.const(32,dtype="float64")
    # constant10=relay.const(32,dtype="int32")
    # constant11=relay.const(32,dtype="int32")
    # constant12=relay.const(32,dtype="int32")
    # constant13=relay.const(32,dtype="int32")
    # data5 = _expr.Tuple((data4,constant7,constant8,constant9,constant10,constant11,constant12,constant13))
    
    data6 = dl.relay.op.qnn.quantized_conv2d(
        [data4, _expr.const(constant7),_expr.const(constant8)],
        channels=32,
        kernel_size=[1, 1],
        data_layout="NCHW",
        out_dtype="float16", 
        pad_width=[[0, 0], [0, 0], [0, 0], [0, 0]],
        data_scales=[0.0625],
        data_zero_points=[0],
        weight_scales=[0.00860167, 0.0112443, 0.00354004, 0.00978661, 0.00312328, 0.00377464,
                    0.0104451, 0.00250959, 0.00600719, 0.00182056, 0.0199561, 0.00570774, 
                    0.00166702, 0.00889587, 0.00189066, 0.00350809, 0.00461531, 0.00316668,
                    0.00275946, 0.00676012, 0.00263548, 0.00974369, 0.00397301, 0.00383091, 
                    0.00387096, 0.00810099, 0.00851011, 0.00363541, 0.00362539, 0.0110407,
                    0.00447512, 0.0051074], 
        weight_zero_points=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        scales=[0.0625], 
        zero_points=[0])
    
    data7 = dl.relay.op.qnn.quantize(data3,out_dtype="int8", output_scales=[0.0625], output_zero_points=[0])

    # constant14=relay.zeros((32,32,3,3),dtype="int8")
    constant14=np.zeros((32,32,3,3)).astype('int8')
    constant15=np.ones(32,).astype('int32')

    # constant15=relay.const(32,dtype='int32') 
    # constant16=relay.const(32,dtype="float64")
    # constant17=relay.const(32,dtype="int32")
    # constant18=relay.const(32,dtype="int32")
    # constant19=relay.const(32,dtype="int32")
    # constant20=relay.const(32,dtype="int32")
    # data8 = _expr.Tuple((data7,constant14,constant15,constant16,constant17,constant18,constant19,constant20))

    data9 = dl.relay.op.qnn.quantized_conv2d(
        [data7,_expr.const(constant14),_expr.const(constant15)],
        channels=32, 
        kernel_size=(3, 3), 
        data_layout="NCHW",
        out_dtype="float16",
        pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), 
        data_scales=[0.0625],
        data_zero_points=[0],
        weight_scales=[ 0.000747681, 0.00150204, 0.00254524, 0.00138867, 0.00198698, 0.00314009,
                        0.00226021, 0.00310171, 0.00090003, 0.000957251, 0.00198185, 0.00095892, 
                        0.00102282, 0.000946045, 0.00104773, 0.00133944, 0.0036273, 0.00482547,
                        0.00132942, 0.000812054, 0.00244832, 0.00148559, 0.00139821, 0.00524104,
                        0.00255167, 0.00317216, 0.00236082, 0.000531793, 0.0013566, 0.00268483,
                        0.000910401, 0.00435686],
        weight_zero_points=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        scales=[0.0625],
        zero_points=[0]
            )
    
    data10 = relay.nn.relu(data6)
    data11 = relay.nn.relu(data9)
    data12 = _expr.Tuple((data10, data11))
    data13 = relay.concatenate(data12, axis=1) # ???
    data14 = dl.relay.op.qnn.quantize(data13,out_dtype="int8", output_scales=[0.0625], output_zero_points=[0])

    # constant21=relay.zeros((65,64,1,1),dtype="int8")
    constant21=np.zeros((65,64,1,1)).astype('int8')
    constant22=np.ones(65,).astype('int32')
    # constant22=relay.const(65,dtype='int32') 
    # constant23=relay.const(65,dtype="float64")
    # constant24=relay.const(65,dtype="int32")
    # constant25=relay.const(65,dtype="int32")
    # constant26=relay.const(65,dtype="int32")
    # constant27=relay.const(65,dtype="int32")
    # data15 = _expr.Tuple((data14,constant21,constant22,constant23,constant24,constant25,constant26,constant27))
    data153_var = dl.relay.op.qnn.quantized_conv2d(
        [data14, _expr.const(constant21), _expr.const(constant22)],
        channels=65, 
        kernel_size=(1, 1), 
        data_layout="NCHW", 
        out_dtype="float16", 
        pad_width=((0, 0), (0, 0), (0, 0), (0, 0)), 
        data_scales=[0.0625], 
        data_zero_points=[0], 
        weight_scales=[0.00724363, 0.00503778, 0.00103235, 0.00153065, 0.00787449, 0.0031147, 
                    0.013937, 0.00413513, 0.00294781, 0.00963497, 0.0021081, 0.00210381, 
                    0.00330114, 0.00659704, 0.00509071, 0.00111723, 0.001194, 0.00875473, 
                    0.0109134, 0.0100994, 0.00987482, 0.00547409, 0.0135169, 0.00549412, 
                    0.00582743, 0.00884485, 0.00637388, 0.00533295, 0.000562191, 0.000825405, 
                    0.0080719, 0.0126028, 0.0102587, 0.0146933, 0.00774431, 0.0151076, 
                    0.00844193, 0.0080471, 0.0149021, 0.00746441, 0.00580072, 0.00137138, 
                    0.00178576, 0.00935602, 0.0115585, 0.0105648, 0.0169311, 0.0107989, 
                    0.0140147, 0.0102644, 0.0098877, 0.0154767, 0.00502348, 0.00725126, 
                    0.00125837, 0.00180101, 0.00978327, 0.00679588, 0.00757647, 0.0104213, 
                    0.0109987, 0.00825548, 0.0071764, 0.00620747, 0.00928736], 
        weight_zero_points=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        scales=[0.25], 
        zero_points=[0]
        )
    
    print("---test_hk_nms_all_plugin---")
    # const226 = 20

    # 1. need specify
    const227 = relay.ones((5,2), dtype="float32")
    # const228 = 5
    # const228 = 40

    # 2. need to specify im_info_var
    im_info_var = relay.ones((1,3),dtype="float32")

    # 3. need to specify data153_var
    # data153_var = relay.ones((1, 65, 20, 32),dtype="float32")
 
    # 4. need to specify how to take
    data154 = relay.take(data153_var, indices=relay.arange(relay.const(0),relay.const(20),dtype="int32"), axis=1, batch_dims=0, mode="clip")
    data155 = relay.transpose(data154, (0, 2, 3, 1))
    im_info156 = relay.divide(im_info_var, relay.const(16, dtype="float32"))
    data157 = relay.round(im_info156)
    data158 = relay.cast(data155, "float32")
    data159 = relay.cast(data157, "int32")
    data160 = relay.cast(im_info_var, "int32")

    data161_boxes = tvm.get_global_func("dl.relay.op._make.custom_boxes_plugin") (data158, const227, data159, data160, "NHWC", 0)
    data162 = relay.take(data153_var, indices=relay.arange(relay.const(0),relay.const(5),dtype="int32"), axis=1, mode="fast")
    data163 = relay.take(data153_var, indices=relay.arange(relay.const(25),relay.const(65),dtype="int32"), axis=1, mode="fast")
    data164 = relay.reshape(data163,newshape=[1, 5, 8, 20, 32]) #8x3200
    data165 = relay.transpose(data164, axes=[0, 2, 1, 3, 4])
    data166 = relay.nn.softmax(data165, axis=1)
    data167 = relay.sigmoid(data162) #1x5x20x32
    data168 = relay.reshape(data166,newshape=[1, 40, 20, 32]) # 8x3200
    data169 = _expr.Tuple((data167, data168))

    iou_threshold = 0.45
    score_threshold = 0.2
    keepTopK = -1
    score_threshold_scalar = float(score_threshold)

    import math
    log2block = int(math.ceil(math.log2(3200)))
    total_length = pow(2, log2block)

    # 5. need specify how to use "custom_filter_sort"
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
    data174_sort_size = relay.reshape( data173_sort_size, newshape = [1, 8])   #/ *ty = Tensor[(1, 8), int32] * /;
    data175_boxes = relay.reshape( data161_boxes, newshape = [1, 1, -1, 4])    #/ *ty = Tensor[(1, 1, 3200, 4), float32] * /;

    data177_boxes_gather = tvm.get_global_func( "dl.relay.op._make.custome_non_max_suppression_gather_boxes")(data175_boxes, data172_sorted_idx, data174_sort_size)
    data178_sorted_scores = relay.reshape( data176_sorted_scores, newshape = [1, 8, -1]) #/ *ty = Tensor[(1, 8, 4096), float32] * /;
    iou_threshold_var = relay.const(np.array([iou_threshold]).astype(dtype), dtype=dtype)
    score_threshold_var = relay.const(np.array([score_threshold]).astype(dtype), dtype=dtype)
    max_output_size_per_class_var = relay.const(np.array([keepTopK]).astype("int32"), dtype="int32")

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
    data188 = relay.cast( data184, dtype="float32")    #/ *ty = Tensor[(9999, 2), float32] * /;
    data189 = relay.cast(data186, dtype="float32")     #/ *ty = Tensor[(9999, 1), float32] * /;
    data190 = relay.cast(data187, dtype="float32")     #/ *ty = Tensor[(9999, 4), float32] * /;
    data191 = _expr.Tuple((data188, data189, data190))
    data192 = relay.cast(data172_sorted_idx, dtype="float32") #/ *ty = Tensor[(1, 8, 4096), float32] * /;
    data193 = relay.cast(data174_sort_size, dtype="float32")  #/ *ty = Tensor[(1, 8), float32] * /;
    data194 = relay.concatenate(data191, axis=1)              #/ *ty = Tensor[(9999, 7), float32] * /;
    data195 = _expr.Tuple((data175_boxes, data192, data193, data194))
    output = _expr.Tuple((data193, data188, data189, data190))
    

    # im_info_data = np.zeros((1,3),dtype = np.int32)
    # data153_data = np.zeros((1,65,20,32),dtype = np.float16)

    params = {
        # "im_info": im_info_data,
        # "data153": im_info_data,
    }
    
    net = relay.Function([], output)
    # mod = tvm.IRModule()
    # mod["main"] = net
    # # print(mod)

    mod=tvm.IRModule().from_expr(net)
    print(mod)
    print('...........mod optimize.........')
    mod = dl.relay.ir_pass.optimize(mod)
    json_mod = tvm.ir.save_json(mod)
    file_name = "hktestnms_ok.rlym"
    with open(file_name, "w+") as f:
        f.write(json_mod)

    # return
    output = evaluate_with_nne(mod, params,"0", serialize_file=None, deseralize=False)
    for i in np.arange(len(output)):
        print(f"{output[i].shape}")

    np.save('./output2.npy',output)
    # tmp = np.load('./output.npy', allow_pickle= True)
    # pdb.set_trace()
    # np.allclose(output[2], tmp[2], rtol=1.e-5, atol=1.e-8, equal_nan=False)
    # print("end")
def here():
    return


if __name__ == '__main__':
    test_hk_nms_all_plugin()
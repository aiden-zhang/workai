from __future__ import absolute_import, print_function
import tvm

from tvm import relay
import numpy as np

import dl


def evaluate_with_nne(mod, inputs_dict, config_key="0123"):
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

    # if isinstance(mod["main"].body.checked_type, tvm.ir.type.TupleType):
    #     out_tensors = mod["main"].body.checked_type.fields
    #     outputs_shape_dict = {
    #         "out_{}".format(idx): get_const_tuple(tensor.shape)
    #         for idx, tensor in enumerate(out_tensors)
    #     }
    #     outputs_name=["out_{}".format(idx) for idx in range(len(out_tensors))]
    #
    # else:
    #     out_shape = get_const_tuple(mod["main"].body.checked_type.shape)
    #
    #     outputs_shape_dict = {"out_0": out_shape}
    #     outputs_name = ["out_0"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rlym") as f:
        f.write(tvm.ir.save_json(mod))
        f.flush()

        with nne.Builder() as builder, nne.Parser() as parser:
            weight_mode = weight_share_configs[config_key]["weight_mode"]
            cluster_cfg = weight_share_configs[config_key]["cluster_cfg"]
            print("weight_mode: %s, cluster_cfg: %s" % (weight_mode, cluster_cfg))

            network = builder.create_network()
            builder.config.ws_mode = weight_mode
            builder.config.max_batch_size = 1

            # [parser.register_output(key) for key, value in outputs_shape_dict.items()]

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
                    binding_shape = engine.get_binding_shape(index)
                    # if binding_name in inputs_dict:
                    #     binding_shape = inputs_dict[binding_name].shape
                    #     context.set_bindings_shape(index, binding_shape)
                    # else:
                    #     binding_shape = engine.get_binding_shape(binding_name)

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

                for index in range(nb):
                    if engine.binding_is_input(index) is False:
                        outputs.append(
                            cuda.from_device(
                                bindings[index].mem, batch_input_shape[index], bindings[index].dtype
                            )
                        )

                assert len(outputs)
                return outputs

            outputs = _evaluate(inputs_dict)

            return outputs[0] if len(outputs) == 1 else outputs


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
def run_mod(mod, params, runtime="vm", open_opt=False):
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

        return evaluate_with_nne(mod, params)


def run_onnx_test(model, inputs, outputs=[]):
    if len(outputs) > 0:
        import onnx
        import onnx_extractor

        model = onnx.load_model(model)
        input_names = [i.name for i in model.graph.input]

        extractor = onnx_extractor.Extractor(model)

        extracted = extractor.extract_model(input_names, outputs)

        onnx.save_model(extracted, "./subgraph.onnx")
        model = "./subgraph.onnx"
    import onnxruntime.backend

    rep = onnxruntime.backend.prepare(model, "CPU")
    ref = rep.run(inputs)

    return ref

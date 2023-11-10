import tvm
import dl
from dl.relay.op.transform import non_max_suppression as dl_non_max_suppression
from dl.relay.op.transform import csum
from dl.relay.op.transform import combine_non_max_suppression_post
from dl.relay.op.transform import combine_non_max_suppression_pile_classes
from dl.relay.op.transform import non_max_suppression_gather_boxes
from dl.op import register_tensorflow_converter, overload_tensorflow_converter
from tvm.topi.utils import get_const_int, get_const_tuple
from tvm.relay import op as _op, expr as _expr, function as _function
from tvm.relay import analysis
import dleol
from dleol.relay.op.nn import filter_sort
from tvm.relay.dataflow_pattern import (
    is_op,
    is_tuple,
    is_tuple_get_item,
    wildcard,
    _DFPatternCallback,
    DFPatternCallback,
    ConstantPattern,
    WildcardPattern,
    TupleGetItemPattern,
    rewrite,
)
from dl.relay.pattern import DL_ELEMENT_SUBGRAPH_CALLBACKS,CUSTOM_REWRITE_CALLBACKS

import os



KERNEL_PATH=os.path.dirname(__file__)

KERNEL_PATH=os.path.join(KERNEL_PATH,"./dlnne_plugin/plugin/kernel")

if ""==os.getenv("YOLOV2_PLUGIN_KERNEL_PATH",""):
    os.environ["YOLOV2_PLUGIN_KERNEL_PATH"]=KERNEL_PATH


import dl
from dl import op


TVM_TVM_REGISTER_SO_NAME = os.path.join(os.path.dirname(__file__),"./dlnne_plugin_build/libdlnne_tvm_plugin.so")

op.load_op_library(TVM_TVM_REGISTER_SO_NAME)


class CustomFilterSortSingleInputFusedCallback(DFPatternCallback):

    def __init__(self):
        super().__init__()
        self.x467=wildcard()
        self.x472=wildcard()

        self.x468 = is_op("reshape")(self.x467)
        self.x473 = is_op("reshape")(self.x472)
        self.x476=is_op("reshape")(is_op("transpose")(is_op("multiply")(self.x468,self.x473)))


        self.cast =is_op("cast")(self.x476)
        self.reshape=is_op("reshape")(self.cast)
        self.pattern =is_op("dl.custom_filter_sort_single_input")(is_tuple([self.reshape]))#.has_attr({"axis":0})

    def callback(self, pre, post, node_map):

        x467 = node_map[self.x467][0]
        x472 = node_map[self.x472][0]

        cast = node_map[self.cast][0]

        filter_sort = node_map[self.pattern][0]
        is_ascend = filter_sort.attrs.is_ascend
        threshold = filter_sort.attrs.threshold
        output_shapes = filter_sort.attrs.output_shapes
        Tindices = filter_sort.attrs.Tindices
        Tout = cast.attrs.dtype

        return tvm.get_global_func("dl.relay.op._make.custom_filter_sort_single_input")(_expr.Tuple([x467,x472]),
                    threshold,is_ascend,Tout,Tindices,output_shapes)

CUSTOM_REWRITE_CALLBACKS.append(CustomFilterSortSingleInputFusedCallback())



def _infer_shape(out, mod):
    if isinstance(out, _expr.Var):
        return get_const_tuple(out.type_annotation.shape)
    """A method to get the output shape of intermediate nodes in the relay graph."""
    return tvm.relay.frontend.common.infer_shape(out, mod)

def _infer_value(input_val, params, mod):
    try:
        from tvm.contrib import graph_runtime

        # Check that all free variables have associated parameters.
        assert all(
            var.name_hint in params.keys() for var in analysis.free_vars(input_val)
        ), "All inputs to infer must be available in params."
        func = _function.Function(analysis.free_vars(input_val), input_val)
        target = "llvm"
        ctx = tvm.context(target, 0)
        with tvm.transform.PassContext(opt_level=0):
            graph, lib, params = tvm.relay.build(func, target=target, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**params)
        m.run()
        return m.get_output(0)
    except Exception:
        if isinstance(mod, tvm.IRModule):
            mod["main"] = _function.Function(analysis.free_vars(input_val), input_val)
        else:
            mod = tvm.IRModule.from_expr(input_val)
        exc = tvm.relay.create_executor("debug", mod=mod, ctx=tvm.cpu(), target="llvm")
        inputs = []
        for param in mod["main"].params:
            inputs.append(params[param.name_hint])
        result = exc.evaluate()(*inputs)
        return result



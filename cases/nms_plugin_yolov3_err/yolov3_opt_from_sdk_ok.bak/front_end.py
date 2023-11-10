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


class CustomFilterSortFusedCallback(DFPatternCallback):

    def __init__(self):
        super().__init__()
        self.x467=wildcard()
        self.x472=wildcard()

        self.x468 = is_op("reshape")(self.x467)
        self.x473 = is_op("reshape")(self.x472)
        self.x476=is_op("reshape")(is_op("transpose")(is_op("multiply")(self.x468,self.x473)))

        self.x480=wildcard()
        self.x485=wildcard()
        self.x481 =is_op("reshape")(self.x480)
        self.x486 = is_op("reshape")(self.x485)
        self.x489=is_op("reshape")(is_op("transpose")(is_op("multiply")(self.x481,self.x486)))

        self.x493=wildcard()
        self.x498=wildcard()
        self.x494 = is_op("reshape")(self.x493)
        self.x499=is_op("reshape")(self.x498)
        self.x502=is_op("reshape")(is_op("transpose")(is_op("multiply")(self.x494,self.x499)))


        self.x504 = is_op("concatenate")(is_tuple([self.x476,self.x489,self.x502])).has_attr({"axis":2})

        self.cast =is_op("cast")(self.x504)
        self.reshape=is_op("reshape")(self.cast)
        self.pattern =is_op("dl.custom_filter_sort")(is_tuple([self.reshape]))#.has_attr({"axis":0})

    def callback(self, pre, post, node_map):


        x467 = node_map[self.x467][0]
        x472 = node_map[self.x472][0]

        x480 = node_map[self.x480][0]
        x485 = node_map[self.x485][0]

        x493 = node_map[self.x493][0]
        x498 = node_map[self.x498][0]

        cast = node_map[self.cast][0]

        filter_sort = node_map[self.pattern][0]
        is_ascend = filter_sort.attrs.is_ascend
        threshold = filter_sort.attrs.threshold
        output_shapes = filter_sort.attrs.output_shapes
        Tindices = filter_sort.attrs.Tindices
        Tout = cast.attrs.dtype

        return tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(_expr.Tuple([x467,x472,x480,x485,x493,x498]),
                    threshold,is_ascend,Tout,Tindices,output_shapes)

CUSTOM_REWRITE_CALLBACKS.append(CustomFilterSortFusedCallback())



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


@overload_tensorflow_converter("DlNonMaxSuppression")
def _DlNonMaxSuppression(inputs, attr, params, mod):
    """
    :param inputs:[boxes,scores,max_output_size_per_class,max_total_size,iou_threshold,score_threshold]
    :param attr:
    :param params:
    :param mod:
    :return:
    """

    boxes = inputs[0]
    scores = inputs[1]
    max_output_size_per_class = inputs[2]
    # max_total_size=inputs[3]
    iou_threshold = inputs[4]
    score_threshold = inputs[5]

    score_threshold_scalar = float(_infer_value(inputs[5], params, mod).asnumpy())

    # todo open
    from tvm import relay

    raw_shape = get_const_tuple(_infer_shape(scores, mod))
    boxes_shape = get_const_tuple(_infer_shape(boxes, mod))
    batch=boxes_shape[0]

    import math


    log2block = int(math.ceil(math.log2(int(raw_shape[-1]))))
    total_length = pow(2, log2block)

    if len(raw_shape)!=2:
        scores_reshape=relay.reshape(scores,[-1,raw_shape[-1]])
        pre_s=1
        for s in raw_shape[:-1]:
            pre_s*=s
    else:
        pre_s=raw_shape[0]
        scores_reshape=scores

        raw_shape=[batch,pre_s//batch]


    sorted_scores_idx = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        _expr.Tuple([scores_reshape]),
        score_threshold_scalar,
        False,
        "float32",
        "int32",
        [[pre_s]+[total_length],[pre_s]+[total_length]]
        )



    sorted_scores, sorted_idx, sort_size = _expr.TupleWrapper(sorted_scores_idx, 3)

    sorted_scores = relay.reshape(sorted_scores, list(raw_shape[:-1]) + [-1])
    sorted_idx = relay.reshape(sorted_idx, list(raw_shape[:-1]) + [-1])
    sort_size = relay.reshape(sort_size, list(raw_shape[0:2]))


    boxes_gather = tvm.get_global_func("dl.relay.op._make.custome_non_max_suppression_gather_boxes")(boxes, sorted_idx, sort_size)

    boxes_ids_count = dl.op.custom_non_max_suppression(boxes_gather,
            sorted_scores,
            max_output_size_per_class,
            iou_threshold,
            score_threshold,
            sort_size, 0, "int32", 1)

    selected_ids, count, sort_size = _expr.TupleWrapper(boxes_ids_count, 3)


    csum_value = tvm.get_global_func("dl.relay.op._make.custom_csum")(count,-1,1)


    func=tvm.get_global_func("dl.relay.op._make.custom_combine_non_max_suppression_post")
    l1 = func(
        boxes_gather, sorted_scores, selected_ids, count, csum_value,9999
    )



    outs = _expr.TupleWrapper(l1, 4)
    return outs


@overload_tensorflow_converter("DLBoxes")
def _DLBoxes_(inputs, attr, params, mod):
    """
    :param inputs:Tuple([None,H,W,NUM_ANCHORS*4],[None,2],[2]/[4],[2]/[None,2])
    :param data_format:NHWC/NCHW
    :return:(None,H,W,NUM_ANCHORS,4)/
            (None,NUM_ANCHORS,4,H,W)
    """

    feats = inputs[0]
    anchors = inputs[1]
    input_shape = inputs[2]
    images_shape = inputs[3]
    data_format = attr["data_format"] if "data_format" in attr.keys() else "NHWC"
    align_out = attr["align_out"] if "align_out" in attr.keys() else 0

    func=tvm.get_global_func("dl.relay.op._make.custom_boxes_plugin")
    out = func(feats, anchors, input_shape, images_shape, data_format, align_out)

    return out
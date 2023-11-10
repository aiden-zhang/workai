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
from tvm.target import override_native_generic_func
from tvm.relay.op.strategy.generic import wrap_topi_schedule
from tvm import relay
import numpy as np
from tvm import te
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay.frontend.common import infer_type as _infer_type

from tvm.relay.frontend.onnx import (
    OnnxOpConverter,
    _get_convert_map,
)

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



def _infer_shape_tf(out, mod):
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

    raw_shape = get_const_tuple(_infer_shape_tf(scores, mod))
    boxes_shape = get_const_tuple(_infer_shape_tf(boxes, mod))
    if len(boxes_shape) == 3:
        boxes = relay.reshape(boxes,[boxes_shape[0],1,boxes_shape[1],boxes_shape[2]])
        boxes_shape = get_const_tuple(_infer_shape_tf(boxes, mod))
    if boxes_shape[2] != raw_shape[2]:
        scores = relay.transpose(scores,[0,2,1])
        raw_shape = get_const_tuple(_infer_shape_tf(scores, mod))

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

def custom_filter_sort_op(attrs, inputs, out_types):
    data_buf = tvm.tir.decl_buffer(inputs[0].shape, inputs[0].dtype, "data_buf")
    
    outputs = (
        [out_types] if isinstance(out_types, tvm.ir.type.TupleType) is False else out_types.fields
    )
    out_buf0 = tvm.tir.decl_buffer(outputs[0].shape, outputs[0].dtype, "data_buf0")
    out_buf1 = tvm.tir.decl_buffer(outputs[1].shape, outputs[1].dtype, "data_buf1")
    out_buf2 = tvm.tir.decl_buffer(outputs[2].shape, outputs[2].dtype, "data_buf2")

    out_shape0 = [int(i) for i in outputs[0].shape]
    out_shape1 = [int(i) for i in outputs[1].shape]
    out_shape2 = [int(i) for i in outputs[2].shape]
    output_dtypes = [output.dtype for output in outputs]
    
    out = te.extern(
        [out_shape0,out_shape1,out_shape2],
        [inputs[0]],
        lambda ins, outs: tvm.tir.call_packed(
            "dl.caffe.custom_filter_sort_cpu",
            ins[0],
            outs[0]
        ),
        dtype=output_dtypes,
        in_buffers=[data_buf],
        out_buffers=[out_buf0,out_buf1,out_buf2],
        name="custom_filter_sort_cpu",
    )
    if isinstance(out, tvm.te.tensor.Tensor):
        return [out]
    else:
        return out


def schedule_custom_filter_sort(outs):
    return te.create_schedule([x.op for x in outs])


def wrap_compute_custom_filter_sort(topi_compute):
    def _compute_wrap_compute_custom_filter_sort(attrs, inputs, out_type):
        return topi_compute(
           attrs,
           inputs,
           out_type
        )

    return _compute_wrap_compute_custom_filter_sort


@override_native_generic_func("custom_filter_sort_strategy")
def custom_filter_sort_strategy(attrs, inputs, out_type, target):
    """generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_custom_filter_sort(custom_filter_sort_op),
        wrap_topi_schedule(schedule_custom_filter_sort),
        name="custom_filter_sort.generic",
    )
    return strategy


_op.op.register_strategy("dl.custom_filter_sort", custom_filter_sort_strategy)


def custome_non_max_suppression_gather_boxes_op(attrs, inputs, out_types):
    data_buf = tvm.tir.decl_buffer(inputs[0].shape, inputs[0].dtype, "data_buf")
    
    outputs = (
        [out_types] if isinstance(out_types, tvm.ir.type.TupleType) is False else out_types.fields
    )
    out_buf0 = tvm.tir.decl_buffer(outputs[0].shape, outputs[0].dtype, "data_buf0")
    

    out_shape0 = [int(i) for i in outputs[0].shape]
    
    output_dtypes = [output.dtype for output in outputs]
    
    out = te.extern(
        [out_shape0],
        [inputs[0]],
        lambda ins, outs: tvm.tir.call_packed(
            "dl.caffe.custome_non_max_suppression_gather_boxes_cpu",
            ins[0],
            outs[0]
        ),
        dtype=output_dtypes,
        in_buffers=[data_buf],
        out_buffers=[out_buf0],
        name="custome_non_max_suppression_gather_boxes_cpu",
    )
    if isinstance(out, tvm.te.tensor.Tensor):
        return [out]
    else:
        return out


def schedule_custome_non_max_suppression_gather_boxes(outs):
    return te.create_schedule([x.op for x in outs])


def wrap_compute_custome_non_max_suppression_gather_boxes(topi_compute):
    def _compute_wrap_compute_custome_non_max_suppression_gather_boxes(attrs, inputs, out_type):
        return topi_compute(
           attrs,
           inputs,
           out_type
        )

    return _compute_wrap_compute_custome_non_max_suppression_gather_boxes


@override_native_generic_func("custome_non_max_suppression_gather_boxes_strategy")
def custome_non_max_suppression_gather_boxes_strategy(attrs, inputs, out_type, target):
    """generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_custome_non_max_suppression_gather_boxes(custome_non_max_suppression_gather_boxes_op),
        wrap_topi_schedule(schedule_custome_non_max_suppression_gather_boxes),
        name="custome_non_max_suppression_gather_boxes.generic",
    )
    return strategy


_op.op.register_strategy("dl.custome_non_max_suppression_gather_boxes", custome_non_max_suppression_gather_boxes_strategy)


def custom_non_max_suppression_op(attrs, inputs, out_types):
    data_buf = tvm.tir.decl_buffer(inputs[0].shape, inputs[0].dtype, "data_buf")
    
    outputs = (
        [out_types] if isinstance(out_types, tvm.ir.type.TupleType) is False else out_types.fields
    )
    out_buf0 = tvm.tir.decl_buffer(outputs[0].shape, outputs[0].dtype, "data_buf0")
    out_buf1 = tvm.tir.decl_buffer(outputs[1].shape, outputs[1].dtype, "data_buf0")
    out_buf2 = tvm.tir.decl_buffer(outputs[2].shape, outputs[2].dtype, "data_buf0")
    

    out_shape0 = [int(i) for i in outputs[0].shape]
    out_shape1 = [int(i) for i in outputs[1].shape]
    out_shape2 = [int(i) for i in outputs[2].shape]
    
    output_dtypes = [output.dtype for output in outputs]
    
    out = te.extern(
        [out_shape0,out_shape1,out_shape2],
        [inputs[0]],
        lambda ins, outs: tvm.tir.call_packed(
            "dl.caffe.custom_non_max_suppression_cpu",
            ins[0],
            outs[0]
        ),
        dtype=output_dtypes,
        in_buffers=[data_buf],
        out_buffers=[out_buf0,out_buf1,out_buf2],
        name="custom_non_max_suppression_cpu",
    )
    if isinstance(out, tvm.te.tensor.Tensor):
        return [out]
    else:
        return out


def schedule_custom_non_max_suppression(outs):
    return te.create_schedule([x.op for x in outs])


def wrap_compute_custom_non_max_suppression(topi_compute):
    def _compute_wrap_compute_custom_non_max_suppression(attrs, inputs, out_type):
        return topi_compute(
           attrs,
           inputs,
           out_type
        )

    return _compute_wrap_compute_custom_non_max_suppression


@override_native_generic_func("custom_non_max_suppression_strategy")
def custom_non_max_suppression_strategy(attrs, inputs, out_type, target):
    """generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_custom_non_max_suppression(custom_non_max_suppression_op),
        wrap_topi_schedule(schedule_custom_non_max_suppression),
        name="custom_non_max_suppression.generic",
    )
    return strategy


_op.op.register_strategy("custom_non_max_suppression", custom_non_max_suppression_strategy)

def custom_csum_op(attrs, inputs, out_types):
    data_buf = tvm.tir.decl_buffer(inputs[0].shape, inputs[0].dtype, "data_buf")
    
    outputs = (
        [out_types] if isinstance(out_types, tvm.ir.type.TupleType) is False else out_types.fields
    )
    out_buf0 = tvm.tir.decl_buffer(outputs[0].shape, outputs[0].dtype, "data_buf0")
    # out_buf1 = tvm.tir.decl_buffer(outputs[1].shape, outputs[1].dtype, "data_buf0")
    # out_buf2 = tvm.tir.decl_buffer(outputs[2].shape, outputs[2].dtype, "data_buf0")
    

    out_shape0 = [int(i) for i in outputs[0].shape]
    # out_shape1 = [int(i) for i in outputs[1].shape]
    # out_shape2 = [int(i) for i in outputs[2].shape]
    
    output_dtypes = [output.dtype for output in outputs]
    
    out = te.extern(
        [out_shape0],
        [inputs[0]],
        lambda ins, outs: tvm.tir.call_packed(
            "dl.caffe.custom_csum_cpu",
            ins[0],
            outs[0]
        ),
        dtype=output_dtypes,
        in_buffers=[data_buf],
        out_buffers=[out_buf0],
        name="custom_csum_cpu",
    )
    if isinstance(out, tvm.te.tensor.Tensor):
        return [out]
    else:
        return out


def schedule_custom_csum(outs):
    return te.create_schedule([x.op for x in outs])


def wrap_compute_custom_csum(topi_compute):
    def _compute_wrap_compute_custom_csum(attrs, inputs, out_type):
        return topi_compute(
           attrs,
           inputs,
           out_type
        )

    return _compute_wrap_compute_custom_csum


@override_native_generic_func("custom_csum_strategy")
def custom_csum_strategy(attrs, inputs, out_type, target):
    """generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_custom_csum(custom_csum_op),
        wrap_topi_schedule(schedule_custom_csum),
        name="custom_csum.generic",
    )
    return strategy


_op.op.register_strategy("dl.custom_csum", custom_csum_strategy)

def custom_combine_non_max_suppression_post_op(attrs, inputs, out_types):
    data_buf = tvm.tir.decl_buffer(inputs[0].shape, inputs[0].dtype, "data_buf")
    
    outputs = (
        [out_types] if isinstance(out_types, tvm.ir.type.TupleType) is False else out_types.fields
    )
    out_buf0 = tvm.tir.decl_buffer(outputs[0].shape, outputs[0].dtype, "data_buf0")
    out_buf1 = tvm.tir.decl_buffer(outputs[1].shape, outputs[1].dtype, "data_buf0")
    out_buf2 = tvm.tir.decl_buffer(outputs[2].shape, outputs[2].dtype, "data_buf0")
    out_buf3 = tvm.tir.decl_buffer(outputs[3].shape, outputs[3].dtype, "data_buf0")
    

    out_shape0 = [int(i) for i in outputs[0].shape]
    out_shape1 = [int(i) for i in outputs[1].shape]
    out_shape2 = [int(i) for i in outputs[2].shape]
    out_shape3 = [int(i) for i in outputs[3].shape]
    
    output_dtypes = [output.dtype for output in outputs]
    
    out = te.extern(
        [out_shape0,out_shape1,out_shape2,out_shape3],
        [inputs[0]],
        lambda ins, outs: tvm.tir.call_packed(
            "dl.caffe.custom_combine_non_max_suppression_post_cpu",
            ins[0],
            outs[0]
        ),
        dtype=output_dtypes,
        in_buffers=[data_buf],
        out_buffers=[out_buf0,out_buf1,out_buf2,out_buf3],
        name="custom_combine_non_max_suppression_post_cpu",
    )
    if isinstance(out, tvm.te.tensor.Tensor):
        return [out]
    else:
        return out


def schedule_custom_combine_non_max_suppression_post(outs):
    return te.create_schedule([x.op for x in outs])


def wrap_compute_custom_combine_non_max_suppression_post(topi_compute):
    def _compute_wrap_compute_custom_combine_non_max_suppression_post(attrs, inputs, out_type):
        return topi_compute(
           attrs,
           inputs,
           out_type
        )

    return _compute_wrap_compute_custom_combine_non_max_suppression_post


@override_native_generic_func("custom_combine_non_max_suppression_post_strategy")
def custom_combine_non_max_suppression_post_strategy(attrs, inputs, out_type, target):
    """generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_custom_combine_non_max_suppression_post(custom_combine_non_max_suppression_post_op),
        wrap_topi_schedule(schedule_custom_combine_non_max_suppression_post),
        name="custom_combine_non_max_suppression_post.generic",
    )
    return strategy


_op.op.register_strategy("dl.custom_combine_non_max_suppression_post", custom_combine_non_max_suppression_post_strategy)

original_get_convert_map = _get_convert_map

def dl_get_convert_map(opset):
    _dl_convert_map = {
        "DLNonMaxSuppression":DLNonMaxSuppression.get_converter(opset),
    }

    convert_map = original_get_convert_map(opset)
    # Override with our convert map
    convert_map.update(_dl_convert_map)
    return convert_map

setattr(tvm.relay.frontend.onnx, "_get_convert_map", dl_get_convert_map)


class DLNonMaxSuppression(OnnxOpConverter):

    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        """
        convert tensorflow CombinedNonMaxSuppression
        :param inputs:[ boxes,scores]
        :param attr:attr:[keepTopK,topK,iou_threshold,score_threshold]
        :param params:
        :param mod:
        :return:
        """
        boxes = inputs[0]
        scores = inputs[1]
        # classes = inputs[2]

        iou_threshold = attr['iouThreshold']
        score_threshold = attr['scoreThreshold']
        num_classes = attr['numClasses']
        top_k = attr['topK']
        keepTopK = attr['keepTopK']
        background_label_id = attr['backgroundLabelId']

        boxes_shape = _infer_shape(boxes)   #1 1 25200 4
        raw_shape = _infer_shape(scores)
        dtype = _infer_type(scores).checked_type.dtype

        score_threshold_scalar = float(score_threshold)

        # todo open
        if int(raw_shape[1]) != int(num_classes): #todo,num_classes get from attr
            scores = relay.transpose(scores,[0,2,1])
        raw_shape = _infer_shape(scores)

        if boxes_shape[1] != 1:
            boxes = relay.reshape(boxes,[boxes_shape[0],1,boxes_shape[1],4])
        boxes_shape = _infer_shape(boxes)

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
        # import pdb
        # pdb.set_trace()
        boxes_gather = tvm.get_global_func( "dl.relay.op._make.custome_non_max_suppression_gather_boxes")(boxes, sorted_idx, sort_size)

        iou_threshold_var = relay.const(np.array([iou_threshold]).astype(dtype), dtype=dtype)
        score_threshold_var = relay.const(np.array([score_threshold]).astype(dtype), dtype=dtype)
        max_output_size_per_class_var = relay.const(np.array([keepTopK]).astype("int32"), dtype="int32")

        # boxes_ids_count = tvm.get_global_func("dl.relay.op._make.custom_non_max_suppression")(boxes_gather,
        #         sorted_scores,
        #         max_output_size_per_class_var,
        #         iou_threshold_var,
        #         score_threshold_var,
        #         sort_size, 0, "int32", 1)
        boxes_ids_count = dl.op.custom_non_max_suppression(boxes_gather,
            sorted_scores,
            max_output_size_per_class_var,
            iou_threshold_var,
            score_threshold_var,
            sort_size, 0, "int32", 1)

        selected_ids, count, sort_size = _expr.TupleWrapper(boxes_ids_count, 3)

        csum_value = tvm.get_global_func("dl.relay.op._make.custom_csum")(count,-1,1)

        func=tvm.get_global_func("dl.relay.op._make.custom_combine_non_max_suppression_post")
        l1 = func(
            boxes_gather, sorted_scores, selected_ids, count, csum_value,9999
        )
        outs = _expr.TupleWrapper(l1, 4)
        return outs


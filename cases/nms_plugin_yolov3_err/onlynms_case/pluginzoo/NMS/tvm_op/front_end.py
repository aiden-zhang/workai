import tvm
import dl
import numpy as np
from tvm.relay import op as _op, expr as _expr
from tvm import relay
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

def get_checked_type(x):

    try:
        return x.checked_type
    except Exception as e:
        import tvm
        return tvm.relay.frontend.common.infer_type(x).checked_type

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
        # import pdb
        # pdb.set_trace()
        # boxes = inputs[0]
        # scores = inputs[1]
        # classes = inputs[2]

        iou_threshold = 0.45#attr['iouThreshold']
        score_threshold = 0.2#attr['scoreThreshold']
        num_classes = attr['numClasses']
        # top_k = attr['topK']
        # keepTopK = attr['keepTopK']
        keepTopK = -1
        background_label_id = attr['backgroundLabelId']
        score_threshold_scalar = float(score_threshold)

        # boxes_shape = _infer_shape(boxes)   #1 1 25200 4
        # raw_shape = _infer_shape(scores)
        # dtype = _infer_type(scores).checked_type.dtype


        # # todo open
        # if int(raw_shape[1]) != int(num_classes): #todo,num_classes get from attr
        #     scores = relay.transpose(scores,[0,2,1])
        # raw_shape = _infer_shape(scores)

        # if boxes_shape[1] != 1:
        #     boxes = relay.reshape(boxes,[boxes_shape[0],1,boxes_shape[1],4])
        # boxes_shape = _infer_shape(boxes)

        # batch=boxes_shape[0]

        # import math

        # log2block = int(math.ceil(math.log2(int(raw_shape[-1]))))
        # total_length = pow(2, log2block)

        # if len(raw_shape)!=2:
        #     scores_reshape=relay.reshape(scores,[-1,raw_shape[-1]])
        #     pre_s=1
        #     for s in raw_shape[:-1]:
        #         pre_s*=s
        # else:
        #     pre_s=raw_shape[0]
        #     scores_reshape=scores

        #     raw_shape=[batch,pre_s//batch]

        # sorted_scores_idx = tvm.get_global_func("dl.relay.op._make.custom_filter_sort")(
        #     _expr.Tuple([scores_reshape]),
        #     score_threshold_scalar,
        #     False,
        #     "float32",
        #     "int32",
        #     [[pre_s]+[total_length],[pre_s]+[total_length]]
        #     )
        
        #sorted_scores 没给，给了boxes box数目不对
        ############## zntest begine
        # sorted_scores, sorted_idx, sort_size = _expr.TupleWrapper(sorted_scores_idx, 3)

        boxes         = inputs[0]
        sorted_scores = inputs[1]
        sorted_idx    = inputs[2]
        sort_size     = inputs[3]
        # raw_shape=?
        dtype="float32"

        # sorted_scores = relay.reshape(sorted_scores, list(raw_shape[:-1]) + [-1])
        # sorted_idx = relay.reshape(sorted_idx, list(raw_shape[:-1]) + [-1])
        # sort_size = relay.reshape(sort_size, list(raw_shape[0:2]))

        ############### zntest end

        boxes_gather = tvm.get_global_func( "dl.relay.op._make.custome_non_max_suppression_gather_boxes")(boxes, sorted_idx, sort_size)

        iou_threshold_var = relay.const(np.array([iou_threshold]).astype(dtype), dtype=dtype)
        score_threshold_var = relay.const(np.array([score_threshold]).astype(dtype), dtype=dtype)
        max_output_size_per_class_var = relay.const(np.array([keepTopK]).astype("int32"), dtype="int32")

        boxes_ids_count = tvm.get_global_func("dl.relay.op._make.custom_non_max_suppression")(boxes_gather,
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


_op.op.register_strategy("dl.custom_non_max_suppression", custom_non_max_suppression_strategy)

def custom_csum_op(attrs, inputs, out_types):
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
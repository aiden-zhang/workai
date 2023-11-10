/*!
 *  Copyright (c) 2018 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
// clang-format off
#include <tvm/ir/error.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/ir/attrs.h>


#include <vector>
// clang-format on

namespace tvm {
namespace relay {
namespace dl {

struct DlNonMaxSuppressionNNEAttrs_
    : public tvm::AttrsNode<DlNonMaxSuppressionNNEAttrs_> {
  int center_point_box;
  DataType Tindices;
  int sorted;
  Array<Array<IndexExpr>> DLExtParamsStrides;
  Array<Array<IndexExpr>> DLExtRetStrides;

  TVM_DECLARE_ATTRS(DlNonMaxSuppressionNNEAttrs_,
                    "relay.attrs.DlNonMaxSuppressionNNEAttrs_") {
    TVM_ATTR_FIELD(center_point_box).describe("center_point_box is 1 or 0...");
    TVM_ATTR_FIELD(Tindices).describe("Tindices is int32/int64...");
    TVM_ATTR_FIELD(sorted).describe("sorted...");
    TVM_ATTR_FIELD(DLExtParamsStrides)
        .set_default(NullValue<Array<Array<IndexExpr>>>())
        .describe("dl_ext_params_strides");
    TVM_ATTR_FIELD(DLExtRetStrides)
        .set_default(NullValue<Array<Array<IndexExpr>>>())
        .describe("dl_ext_params_strides");
  }
};

TVM_REGISTER_NODE_TYPE(DlNonMaxSuppressionNNEAttrs_);
Expr MakeDlNonMaxSuppressionNNE_(Expr boxes_gather, Expr sorted_scores,
                                 Expr max_output_size_per_class,
                                 Expr iou_threshold, Expr score_threshold,
                                 Expr sort_size, int center_point_box,
                                 DataType Tindices, int sorted = 1) {
  auto attrs = make_object<DlNonMaxSuppressionNNEAttrs_>();
  attrs->center_point_box = center_point_box;
  attrs->Tindices = Tindices;
  attrs->sorted = sorted;
  static const Op &op = Op::Get("dl.custom_non_max_suppression");
  return Call(op,
              {boxes_gather, sorted_scores, max_output_size_per_class,
               iou_threshold, score_threshold, sort_size},
              Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("dl.relay.op._make.custom_non_max_suppression").set_body_typed(MakeDlNonMaxSuppressionNNE_);

bool DlNonMaxSuppressionNNERel_(const Array<Type> &types, int num_inputs,
                                const Attrs &attrs,
                                const TypeReporter &reporter) {
  CHECK_EQ(types.size(), 7);
  auto attr = attrs.as<DlNonMaxSuppressionNNEAttrs_>();

  // const auto* data = types[0].as<TupleTypeNode>();
  const auto *scores =
      types[1].as<TensorTypeNode>();  // Downcast<TensorType>(data->fields[1]);
  const auto *boxes =
      types[0].as<TensorTypeNode>();  // Downcast<TensorType>(data->fields[0]);

  const auto &oshape = scores->shape;

  std::vector<Type> fields;
  std::vector<IndexExpr> counts_shape({oshape[0], oshape[1]});

  std::vector<IndexExpr> o_shape({oshape[0], oshape[1], boxes->shape[2]});

  fields.push_back(TensorType(o_shape, attr->Tindices));
  fields.push_back(TensorType(counts_shape, attr->Tindices));
  fields.push_back(TensorType(o_shape, DataType::Bool()));

  reporter->Assign(types[6], TupleType(Array<Type>(fields)));

  return true;
}

RELAY_REGISTER_OP("dl.custom_non_max_suppression")
    .describe(R"code(sorted of a data.
    - **data**: (2-5) boxes,scores,max_output_boxes_per_class(optional tensorScalar)
    ,iou_threshold(optional tensorScalar),score_threshold(optional tensorScalar)

    Example::
      -  dl.custom_non_max_suppression

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(6)

    .set_attrs_type<DlNonMaxSuppressionNNEAttrs_>()
    .add_type_rel("DlNonMaxSuppression_", DlNonMaxSuppressionNNERel_)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

    TVM_REGISTER_GLOBAL("dl.caffe.custom_non_max_suppression_cpu")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    
  });

}  // dl
}  // namespace relay
}  // namespace tvm

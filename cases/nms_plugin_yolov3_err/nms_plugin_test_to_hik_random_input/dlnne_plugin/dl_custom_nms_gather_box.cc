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

// yolo non_max_suppression_gather_boxes
struct DlNonMaxSuppressionGatherBoxesNNEAttrs_
    : public tvm::AttrsNode<DlNonMaxSuppressionGatherBoxesNNEAttrs_> {
  TVM_DECLARE_ATTRS(DlNonMaxSuppressionGatherBoxesNNEAttrs_,
                    "relay.attrs.DlNonMaxSuppressionGatherBoxesNNEAttrs_") {}
};

TVM_REGISTER_NODE_TYPE(DlNonMaxSuppressionGatherBoxesNNEAttrs_);

Expr MakeDlNonMaxSuppressionGatherBoxesNNE_(Expr boxes, Expr sorted_idx,
                                            Expr sort_size) {
  auto attrs = make_object<DlNonMaxSuppressionGatherBoxesNNEAttrs_>();
  static const Op &op = Op::Get("custome_non_max_suppression_gather_boxes");
  return Call(op, {boxes, sorted_idx, sort_size}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.custome_non_max_suppression_gather_boxes")
    .set_body_typed(MakeDlNonMaxSuppressionGatherBoxesNNE_);

bool MakeDlNonMaxSuppressionGatherBoxesNNERel_(const Array<Type> &types,
                                               int num_inputs,
                                               const Attrs &attrs,
                                               const TypeReporter &reporter) {
  //  auto attr = attrs.as<DlNonMaxSuppressionGatherBoxesNNEAttrs_>();
  CHECK_EQ(types.size(), 4);
  // const auto* data = types[0].as<TupleTypeNode>();
  const auto *indices =
      types[1].as<TensorTypeNode>();  // Downcast<TensorType>(data->fields[1]);
  const auto *boxes =
      types[0].as<TensorTypeNode>();  // Downcast<TensorType>(data->fields[0]);
  // const auto sort_size = Downcast<TensorType>(data->fields[2]);

  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < indices->shape.size() - 1; i++) {
    oshape.push_back(indices->shape[i]);
  }
  oshape.push_back(boxes->shape[boxes->shape.size() - 2]);
  oshape.push_back(boxes->shape[boxes->shape.size() - 1]);

  reporter->Assign(types[3], TensorType(oshape, boxes->dtype));
  return true;
}

RELAY_REGISTER_OP("custome_non_max_suppression_gather_boxes")
    .describe(R"code(sorted of a data.
    - **data**: (2) boxes,indices-

    Example::
      -  custome_non_max_suppression_gather_boxes

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(3)

    .set_attrs_type<DlNonMaxSuppressionGatherBoxesNNEAttrs_>()
    .add_type_rel("DlNonMaxSuppressionGatherBoxes_",
                  MakeDlNonMaxSuppressionGatherBoxesNNERel_)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

}  // namespace relay
}  // namespace tvm
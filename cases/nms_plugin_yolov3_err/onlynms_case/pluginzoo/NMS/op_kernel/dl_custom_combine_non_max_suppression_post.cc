#include <tvm/ir/attrs.h>
#include <tvm/ir/error.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <vector>

namespace tvm {
namespace relay {

namespace dl {
struct DlCustomCombineNonMaxSuppressionPostAttrs
    : public tvm::AttrsNode<DlCustomCombineNonMaxSuppressionPostAttrs> {
  IndexExpr max_total_size;

  TVM_DECLARE_ATTRS(DlCustomCombineNonMaxSuppressionPostAttrs,
                    "relay.attrs.DlCustomCombineNonMaxSuppressionPostAttrs") {
    TVM_ATTR_FIELD(max_total_size).set_default(9999).describe("dtype of out");
  }
};

TVM_REGISTER_NODE_TYPE(DlCustomCombineNonMaxSuppressionPostAttrs);
Expr MakeDlCustomCombineNonMaxSuppressionPost(Expr boxes, Expr scores,
                                              Expr selected_idxs, Expr count,
                                              Expr csum,
                                              IndexExpr max_total_size) {
  auto attrs = make_object<DlCustomCombineNonMaxSuppressionPostAttrs>();
  attrs->max_total_size = max_total_size;

  static const Op &op = Op::Get("dl.custom_combine_non_max_suppression_post");
  return Call(op, {boxes, scores, selected_idxs, count, csum}, Attrs(attrs),
              {});
}

TVM_REGISTER_GLOBAL("dl.relay.op._make.custom_combine_non_max_suppression_post")
    .set_body_typed(MakeDlCustomCombineNonMaxSuppressionPost);

bool DlCustomCombineNonMaxSuppressionPostRel(const Array<Type> &types,
                                             int num_inputs, const Attrs &attrs,
                                             const TypeReporter &reporter) {
  auto attr = attrs.as<DlCustomCombineNonMaxSuppressionPostAttrs>();

  const auto *scores = types[1].as<TensorTypeNode>();
  const auto *boxes = types[0].as<TensorTypeNode>();

  auto total_size = attr->max_total_size * boxes->shape[0];

  std::vector<Type> fields;
  std::vector<IndexExpr> idx_shape({total_size, 2});
  std::vector<IndexExpr> boxes_shape({total_size, 4});
  std::vector<IndexExpr> scores_shape({total_size});
  //  std::vector<IndexExpr> len_shape({total_size});

  fields.push_back(TensorType(idx_shape, DataType::Int(32)));
  fields.push_back(TensorType(boxes_shape, boxes->dtype));
  fields.push_back(TensorType(scores_shape, scores->dtype));
  fields.push_back(TensorType({1}, DataType::Int(32)));

  reporter->Assign(types[5], TupleType(Array<Type>(fields)));

  return true;
}
RELAY_REGISTER_OP("dl.custom_combine_non_max_suppression_post")
    .describe(R"code(sorted of a data.
    - **data**: (2-3) boxes_raw_ids,boxes_ids,csum value-

    Example::
      -  DlCustomCombineNonMaxSuppressionPost

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(5)

    .set_attrs_type<DlCustomCombineNonMaxSuppressionPostAttrs>()
    .add_type_rel("DlCustomCombineNonMaxSuppressionPost",
                  DlCustomCombineNonMaxSuppressionPostRel)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

    TVM_REGISTER_GLOBAL("dl.caffe.custom_combine_non_max_suppression_post_cpu").set_body([](TVMArgs args, TVMRetValue* ret) {
    
  });

}  // namespace dl
}  // namespace relay
}  // namespace tvm

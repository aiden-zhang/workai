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
struct DlCustomCsumAttrs : public tvm::AttrsNode<DlCustomCsumAttrs> {
  int only_1dim;
  IndexExpr axis;

  TVM_DECLARE_ATTRS(DlCustomCsumAttrs, "relay.attrs.DlCustomCsumAttrs") {
    TVM_ATTR_FIELD(only_1dim).set_default(1);
    TVM_ATTR_FIELD(axis).set_default(-1);
  }
};

TVM_REGISTER_NODE_TYPE(DlCustomCsumAttrs);
Expr MakeDlCustomCsum(Expr data, int axis, int only_1dim) {
  auto attrs = make_object<DlCustomCsumAttrs>();
  attrs->axis = axis;
  attrs->only_1dim = only_1dim;

  static const Op &op = Op::Get("dl.custom_csum");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("dl.relay.op._make.custom_csum")
    .set_body_typed(MakeDlCustomCsum);

bool DlCustomCsumRel(const Array<Type> &types, int num_inputs,
                     const Attrs &attrs, const TypeReporter &reporter) {
  auto attr = attrs.as<DlCustomCsumAttrs>();

  const auto *data = types[0].as<TensorTypeNode>();

  const auto &oshape = data->shape;

  std::vector<IndexExpr> new_oshape;

  if (attr->only_1dim == 1) {
    IndexExpr size = oshape[0];
    for (size_t i = 1; i < oshape.size(); i++) size = size * oshape[i];

    new_oshape.push_back(size);
    reporter->Assign(types[1], TensorType(new_oshape, data->dtype));
  } else {
    reporter->Assign(types[1], TensorType(oshape, data->dtype));
  }

  return true;
}

RELAY_REGISTER_OP("dl.custom_csum")
    .describe(R"code(sorted of a data.
    - **data**-:

    Example::
      -  dl_csum  -

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<DlCustomCsumAttrs>()
    .add_type_rel("dl_custom_csum", DlCustomCsumRel)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

TVM_REGISTER_GLOBAL("dl.caffe.custom_csum_cpu")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    
  });

}  // namespace dl
}  // namespace relay
}  // namespace tvm

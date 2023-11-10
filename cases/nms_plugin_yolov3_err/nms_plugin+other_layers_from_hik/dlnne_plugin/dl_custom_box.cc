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

struct BoxesAttrsNNE_ : public tvm::AttrsNode<BoxesAttrsNNE_> {
  std::string data_format;
  int align_out;

  TVM_DECLARE_ATTRS(BoxesAttrsNNE_, "relay.attrs.BoxesAttrsNNE_") {
    TVM_ATTR_FIELD(data_format).describe("NCHW/NHWC");
    TVM_ATTR_FIELD(align_out).describe(
        "when align_out==0 [None,None,1,4],[None,None,num_class]"
        "else [None,num_class,None],[None,1,None,4] ");
  }
};

TVM_REGISTER_NODE_TYPE(BoxesAttrsNNE_);

Expr MakeDLBoxesNNE_(Expr data, Expr prior_anchor, Expr input_image_shape,
                     Expr real_image_shape, std::string data_format,
                     int align_out) {
  auto attrs = make_object<BoxesAttrsNNE_>();
  attrs->data_format = data_format;
  attrs->align_out = align_out;

  static const Op &op = Op::Get("dl.custom_boxes_plugin");
  return Call(op, {data, prior_anchor, input_image_shape, real_image_shape},
              Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("dl.relay.op._make.custom_boxes_plugin")
    .set_body_typed(MakeDLBoxesNNE_);

bool DLBoxesRelNNE_(const Array<Type> &types, int num_inputs,
                    const Attrs &attrs, const TypeReporter &reporter) {
  auto attr = attrs.as<BoxesAttrsNNE_>();

  const auto *feats = types[0].as<TensorTypeNode>();
  const auto *anchor = types[1].as<TensorTypeNode>();

  const auto &feats_shape = feats->shape;
  const auto &anchor_shape = anchor->shape;

  std::vector<TensorType> fields;
  IndexExpr b = feats_shape[0];
  IndexExpr h;
  IndexExpr w;
  IndexExpr c;

  IndexExpr num_anchor = anchor_shape[0];
  if (attr->data_format == "NHWC") {
    h = feats_shape[1];
    w = feats_shape[2];
    c = feats_shape[3];
  } else {
    h = feats_shape[2];
    w = feats_shape[3];
    c = feats_shape[1];
  }
  IndexExpr class_num = indexdiv(c, num_anchor) - 5;

  if (attr->data_format == "NCHW") {
    fields.push_back(TensorType({b, num_anchor, 4, h, w}, feats->dtype));
  } else {
    fields.push_back(TensorType({b, h, w, num_anchor, 4}, feats->dtype));
  }

  reporter->Assign(types[4], TensorType(fields[0]));

  return true;
}

RELAY_REGISTER_OP("dl.custom_boxes_plugin")
    .describe(R"code(sorted of a data.
    - **data**-:

    Example::
      -  custom_boxes_plugin  -

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(4)
    .set_attrs_type<BoxesAttrsNNE_>()
    .add_type_rel("custom_boxes_plugin", DLBoxesRelNNE_)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

}  // namespace dl
}  // namespace relay
}  // namespace tvm


#include <tvm/ir/attrs.h>
#include <tvm/ir/error.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {
using namespace tir;

struct DlCustomFilterSortSingleInputAttrs
    : public tvm::AttrsNode<DlCustomFilterSortSingleInputAttrs> {
  IndexExpr threshold;
  DataType Tindices;
  DataType Tout;
  bool is_ascend;
  Array<Array<IndexExpr>> output_shapes;
  IndexExpr class_num;
  IndexExpr count_num;

  Array<Array<IndexExpr>> DLExtParamsStrides;
  Array<Array<IndexExpr>> DLExtRetStrides;
  TVM_DECLARE_ATTRS(DlCustomFilterSortSingleInputAttrs,
                    "relay.attrs.DlCustomFilterSortSingleInputAttrs") {
    TVM_ATTR_FIELD(threshold).describe("threshold");
    TVM_ATTR_FIELD(Tindices).describe("Tindices");
    TVM_ATTR_FIELD(Tout).describe("Tout");

    TVM_ATTR_FIELD(is_ascend).describe("is_ascend");
    TVM_ATTR_FIELD(output_shapes).describe("output_shapes");

    TVM_ATTR_FIELD(class_num).describe("class_num");
    TVM_ATTR_FIELD(count_num).describe("count_num");

    TVM_ATTR_FIELD(DLExtParamsStrides)
        .set_default(NullValue<Array<Array<IndexExpr>>>())
        .describe("dl_ext_params_strides");
    TVM_ATTR_FIELD(DLExtRetStrides)
        .set_default(NullValue<Array<Array<IndexExpr>>>())
        .describe("dl_ext_ret_strides");
  }
};

TVM_REGISTER_NODE_TYPE(DlCustomFilterSortSingleInputAttrs);
Expr MakeDlCustomFilterSortSingleInput(Expr data, IndexExpr threshold, bool is_ascend,
                            DataType Tout, DataType Tindices,
                            Array<Array<IndexExpr>> output_shapes) {
  auto attrs = make_object<DlCustomFilterSortSingleInputAttrs>();
  attrs->threshold = threshold;
  attrs->is_ascend = is_ascend;
  attrs->Tindices = Tindices;
  attrs->Tout = Tout;
  attrs->class_num = output_shapes[0][0];
  attrs->count_num = output_shapes[0][1];

  attrs->output_shapes = output_shapes;

  static const Op &op = Op::Get("dl.custom_filter_sort_single_input");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("dl.relay.op._make.custom_filter_sort_single_input")
    .set_body_typed(MakeDlCustomFilterSortSingleInput);

bool DlCustomFilterSortSingleInputRel(const Array<Type> &types, int num_inputs,
                           const Attrs &attrs, const TypeReporter &reporter) {
  auto attr = attrs.as<DlCustomFilterSortSingleInputAttrs>();

  const auto *datas = types[0].as<TupleTypeNode>();
  auto data = Downcast<TensorType>(datas->fields[0]);

  std::vector<Type> fields;

  fields.push_back(TensorType(attr->output_shapes[0], attr->Tout));
  fields.push_back(TensorType(attr->output_shapes[1], attr->Tindices));
  fields.push_back(TensorType({attr->output_shapes[0][0]}, attr->Tindices));

  reporter->Assign(types[1], TupleType(Array<Type>(fields)));

  return true;
}

RELAY_REGISTER_OP("dl.custom_filter_sort_single_input")
    .describe(R"code(DlCustomFilterSortSingleInput of a data.
    - **data**-:

    Example::
      -  DlCustomFilterSortSingleInputRel  -

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<DlCustomFilterSortSingleInputAttrs>()
    .add_type_rel("dl_custom_filter_sort_single_input", DlCustomFilterSortSingleInputRel)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

}  // namespace relay
}  // namespace tvm

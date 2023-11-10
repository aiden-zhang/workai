
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

struct DlCustomFilterSortAttrs
    : public tvm::AttrsNode<DlCustomFilterSortAttrs> {
  IndexExpr threshold;
  DataType Tindices;
  DataType Tout;
  bool is_ascend;
  Array<Array<IndexExpr>> output_shapes;
  IndexExpr class_num;
  IndexExpr count_num;

  Array<Array<IndexExpr>> DLExtParamsStrides;
  Array<Array<IndexExpr>> DLExtRetStrides;
  TVM_DECLARE_ATTRS(DlCustomFilterSortAttrs,
                    "relay.attrs.DlCustomFilterSortAttrs") {
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

TVM_REGISTER_NODE_TYPE(DlCustomFilterSortAttrs);
Expr MakeDlCustomFilterSort(Expr data, IndexExpr threshold, bool is_ascend,
                            DataType Tout, DataType Tindices,
                            Array<Array<IndexExpr>> output_shapes) {
  auto attrs = make_object<DlCustomFilterSortAttrs>();
  attrs->threshold = threshold;
  attrs->is_ascend = is_ascend;
  attrs->Tindices = Tindices;
  attrs->Tout = Tout;
  attrs->class_num = output_shapes[0][0];
  attrs->count_num = output_shapes[0][1];

  attrs->output_shapes = output_shapes;

  static const Op &op = Op::Get("dl.custom_filter_sort");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("dl.relay.op._make.custom_filter_sort")
    .set_body_typed(MakeDlCustomFilterSort);

bool DlCustomFilterSortRel(const Array<Type> &types, int num_inputs,
                           const Attrs &attrs, const TypeReporter &reporter) {
  auto attr = attrs.as<DlCustomFilterSortAttrs>();

  const auto *datas = types[0].as<TupleTypeNode>();
  auto data = Downcast<TensorType>(datas->fields[0]);

  std::vector<Type> fields;

  fields.push_back(TensorType(attr->output_shapes[0], attr->Tout));
  fields.push_back(TensorType(attr->output_shapes[1], attr->Tindices));
  fields.push_back(TensorType({attr->output_shapes[0][0]}, attr->Tindices));

  reporter->Assign(types[1], TupleType(Array<Type>(fields)));

  return true;
}

RELAY_REGISTER_OP("dl.custom_filter_sort")
    .describe(R"code(DlCustomFilterSort of a data.
    - **data**-:

    Example::
      -  DlCustomFilterSortRel  -

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<DlCustomFilterSortAttrs>()
    .add_type_rel("dl_custom_filter_sort", DlCustomFilterSortRel)
    .set_support_level(1)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false);

TVM_REGISTER_GLOBAL("dl.caffe.custom_filter_sort_cpu")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    
  });

}  // namespace relay
}  // namespace tvm

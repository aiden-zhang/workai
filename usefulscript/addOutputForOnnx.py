import onnx
import numpy as np
import onnx.helper as helper

if __name__=='__main__':
    model = onnx.load('modified_ppyoloe_plus_crn_s_80e_coco.onnx')
    item = model.graph
    # print(f'name: {item.input} input: {item.input} output: {item.output}')

    print("-" * 50)
    # add a output node
    # print(type(output))
    # 1代表数据类型，也可以写成TensorProto.FLOAT
    new_output = helper.make_tensor_value_info('concat_15.tmp_0', 1, [-1, 8400, 4])
    print("-" * 50)
    print(new_output)
    model.graph.output.append(new_output)
    print(model.graph.output)

    for vis in (model.graph.input, model.graph.value_info):
        for vi in vis:
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value == -1:
                    d.dim_value = 1
    #change output shape[0]
    output=model.graph.output[0]
    new_output = helper.make_tensor_value_info(output.name, 1, [1, 8400, 4])
    output.CopyFrom(new_output)

    # add another outputnode
    model.graph.output.append(helper.make_tensor_value_info('concat_14.tmp_0',1,[1,80,8400]))
    onnx.save(model,'new_static.onnx')


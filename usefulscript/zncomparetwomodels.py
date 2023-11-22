from typing import List
import logging
import time
import sys
# sys.path.append(".")
import importlib
from os.path import splitext, basename, split, dirname
from os import getcwd, chdir
import numpy as np
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.shape_parser import parse_shape_string
import dl
# from dl.logging import file_logger
# from ..main import register_parser
from common import get_params_info, load_plugin, create_executor, compare_consine
import argparse
import pdb


# logger = logging.getLogger("dlTVMC")

def eval_acc(
        mod,
        input_data_dict={},
        target="default",
):
    # pdb.set_trace()
    # data = {key: np.ones(inputs_shape_dict[key], dtype='float32') for key in list(inputs_shape_dict)}
    # data = {key: np.load(inputfile)  for key in list(inputs_shape_dict)}

    ex = create_executor(mod, "vm", target)

    # setup evaluation metric
    # dataset.reset()
    # metric.reset()

    # for i, batch in enumerate(dataset):
    #     data = {key: batch[key] for key in list(inputs_shape_dict)}
    #     outs = ex.evaluate(data)
    #     # metric.update(batch["label"], outs)
    #     # if debug_info:
    #     #     message = "{}/{}".format(i, len(dataset))
    #     #     print(message, end="", flush=True)
    #     #     print("\r", end="", flush=True)
    #woyon

    # return metric.get()

    outs = ex.evaluate(input_data_dict)
    return outs


def drive_acc(args):
    # logger.debug("run model={} target={}".format(args.FILE, args.target))

    tvmc_model = tvmc.frontends.load_model(args.model1)
    mod1 = tvmc_model.mod

    tvmc_model = tvmc.frontends.load_model(args.model2)
    mod2 = tvmc_model.mod

    _, inputs_names, inputs_shapes = get_params_info(mod1)

    model_name = splitext(basename(args.model1))[0]
    # module = load_plugin(args.plugin)

    # qPlugIn = module.get_quatization_plugin(
    #     model_name=model_name,
    #     inputs_names=inputs_names,
    #     dataset_path=args.dataset,
    #     common_config=args.common_config,
    # )
    # validate_dataset = qPlugIn.get_validate_dataset(args.num_samples)
    # metric = qPlugIn.get_metric()

    # pdb.set_trace()
    input_shape_dict = {inputs_names[i]: inputs_shapes[i] for i in range(len(inputs_names))}
    input_data_dict = {key: np.load(args.input_data) for key in list(input_shape_dict)}

    output1 = eval_acc(
        mod1,
        input_data_dict,
        target=args.target,
    )

    output2 = eval_acc(
        mod2,
        input_data_dict,
        target=args.target,
    )

    print(f"-----------------------model:{args.model1}, target:{args.target} output1--------------------")
    print(output1)

    print(f"-----------------------model:{args.model2}, target{args.target} output2--------------------")
    print(output2)

    compare_consine(output1, output2)
    # import pdb;pdb.set_trace()
    if(args.dumpOutput):
        np.save('output1.npy', output1)
        np.save('output2.npy', output2)

    return None


if __name__ == '__main__':
    # print('hello world')
    parser = argparse.ArgumentParser(description="compare two models output")
    parser.add_argument("-m1", "--model1", type=str, required=True, default='./feature.rlym', help="assign model1 path including model name")
    parser.add_argument("-m2", "--model2", type=str, required=True, default='./feature_fp32.rlym', help="assign model2 path including model name")
    parser.add_argument('-t', "--target", type=str, default="nne", help="target backend to infer")
    parser.add_argument("--input_data", type=str, default="feature_input2.npy", help="input data of npy")
    parser.add_argument("--dumpOutput", type=bool, default=False, help="dump output data  or not")
    args = parser.parse_args()
    drive_acc(args)


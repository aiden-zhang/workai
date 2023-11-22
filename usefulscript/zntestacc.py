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
from dl.logging import file_logger
# from ..main import register_parser
from common import get_params_info, load_plugin, create_executor


# logger = logging.getLogger("dlTVMC")

def eval_acc(
        mod,
        inputs_shape_dict={},
        target="default",
        debug_info=False,
):
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
    data = {key: np.ones(inputs_shape_dict[key], dtype='float32') for key in list(inputs_shape_dict)}
    outs = ex.evaluate(data)
    return outs


def drive_acc(args):
    # logger.debug("run model={} target={}".format(args.FILE, args.target))

    tvmc_model = tvmc.frontends.load_model(args.FILE)
    mod = tvmc_model.mod

    _, inputs_names, inputs_shapes = get_params_info(mod)

    model_name = splitext(basename(args.FILE))[0]
    # module = load_plugin(args.plugin)

    # qPlugIn = module.get_quatization_plugin(
    #     model_name=model_name,
    #     inputs_names=inputs_names,
    #     dataset_path=args.dataset,
    #     common_config=args.common_config,
    # )
    # validate_dataset = qPlugIn.get_validate_dataset(args.num_samples)
    # metric = qPlugIn.get_metric()
    res = eval_acc(
        mod,
        inputs_shape_dict={inputs_names[i]: inputs_shapes[i] for i in range(len(inputs_names))},
        target=args.target,
    )
    print(res)
    return res


class my_args:
    def __init__(self):
        self.File = None
        self.target = None


if __name__ == '__main__':
    print('hello world')
    args = my_args()
    args.FILE = './justadd.rlym'
    args.FILE = './justadd.onnx'
    args.target = 'llvm'
    args.target = 'onnx'
    args.target = 'nne'

    drive_acc(args)

from typing import Union, List, Tuple, Dict
import re
import time
import sys
import argparse
import numpy as np
import os
from os.path import splitext, basename, dirname, split
from os import getcwd, chdir, environ
import subprocess
import importlib
import importlib.util
import tvm
from tvm import relay
from tvm.relay.analysis.analysis import post_order_visit
from tvm.topi import get_const_tuple
from tvm.driver.tvmc import TVMCException
from dl.relay.backend.executor import NNEEngine, select_target
from scipy import spatial

def _log_duration(start: float, message: str) -> None:
    compile_time = time.time() - start
    if compile_time > 5:
        sys.stderr.write("{} {}(s)\n".format(message, compile_time))


def get_params_info(mod):
    func = mod["main"]
    inputs_name = []
    inputs_shape = []
    inputs_dtype = []
    for v in func.params:
        inputs_name.append(v.name_hint)
        inputs_shape.append(get_const_tuple(v.type_annotation.shape))
        inputs_dtype.append(v.type_annotation.dtype)
    return inputs_dtype, inputs_name, inputs_shape


def guess_input_layout(mod):
    data_layout = None

    def _fvisit(expr):
        nonlocal data_layout
        if data_layout is not None:
            return
        if (
            isinstance(expr, relay.Call)
            and isinstance(expr.op, tvm.ir.Op)
            and expr.op.name in ["dl.quantized_conv2d", "nn.conv2d", "nn.conv3d"]
        ):
            data_layout = expr.attrs.data_layout

    post_order_visit(mod["main"], _fvisit)
    return data_layout


class TVMExecutor(object):
    def __init__(self, executor) -> None:
        super().__init__()
        self.executor = executor

    def evaluate(self, input_dict) -> List[np.ndarray]:
        start = time.time()
        out = self.executor.evaluate()(**input_dict)
        _log_duration(start, "Execution")

        if isinstance(out, tvm.nd.NDArray):
            return [out.numpy()]
        return [o.asnumpy() for o in out]


class ONNXExecutor(object):
    def __init__(self, mod) -> None:
        super().__init__()
        self.mod = mod

    def func_to_onnx(self, mod, params, name):
        from tvm.contrib.target.onnx import to_onnx
        import dl.target.contrib_onnx

        onnx_model = to_onnx(mod, params, name, path=None)
        return onnx_model.SerializeToString()

    def run_onnx(self, mod, params, name, input_data):
        import onnxruntime as rt

        onnx_model = self.func_to_onnx(mod, params, name)
        sess = rt.InferenceSession(onnx_model)
        input_names = {}
        for input, data in zip(sess.get_inputs(), input_data):
            input_names[input.name] = data
        output_names = [output.name for output in sess.get_outputs()]
        res = sess.run(output_names, input_names)
        return res

    def evaluate(self, input_dict) -> List[np.ndarray]:
        start = time.time()
        out = self.run_onnx(self.mod, input_dict, "test", "input_dict")
        _log_duration(start, "Execution")

        return out


def create_executor(
    mod: Union[tvm.IRModule, str],
    backend,
    target,
    hc_profile=False,
    max_batch_size=1,
    cluster_count=1,
    is_create_execution_context=True,
):
    target = select_target(target)

    start = time.time()
    print(f"---target:{target}")
    if target == "nne":
        from tvm.ir.transform import PassContext

        pass_ctx = PassContext.current()
        disabled_pass = pass_ctx.disabled_pass

        assert (
            len(disabled_pass) == 0
        ), "NNEEngine can't pass disabled_pass from PassContext to convert"

        ex = NNEEngine(mod, max_batch_size, cluster_count, is_create_execution_context)
    elif target == "onnx":
        ex = ONNXExecutor(mod)
    else:
        check_unsupported_ops(mod, target)
        ex = relay.create_executor(backend, mod=mod, target=target)
        ex = TVMExecutor(ex)

    _log_duration(start, "Compiling")
    return ex


def from_hist_gen_rand_data_deterministic_contain_min_max(hist, shape):
    cnts = hist[0]
    bounds = hist[1]

    gen_d = np.empty((0,), dtype=bounds.dtype)
    for i, cnt in enumerate(cnts):
        if i == 0 and cnt > 0:
            gen_sub_d = np.random.uniform(bounds[i], bounds[i + 1], size=(cnt - 1,))
            min_value = np.array([bounds[i]], dtype=bounds.dtype)
            gen_sub_d = np.concatenate((min_value, gen_sub_d))
        elif i == len(cnts) - 1 and cnt > 0:
            gen_sub_d = np.random.uniform(bounds[i], bounds[i + 1], size=(cnt - 1,))
            max_value = np.array([bounds[i + 1]], dtype=bounds.dtype)
            gen_sub_d = np.concatenate((gen_sub_d, max_value))
        else:
            gen_sub_d = np.random.uniform(bounds[i], bounds[i + 1], size=(cnt,))
        gen_d = np.concatenate((gen_d, gen_sub_d))
    return gen_d.reshape(shape)


def gen_fake_variable_tensor(hist, shape):
    min = np.min(hist[1])
    max = np.max(hist[1])

    fake_variable_tensor = from_hist_gen_rand_data_deterministic_contain_min_max(hist, shape)

    fake_variable_tensor = np.clip(fake_variable_tensor, min, max)

    flattened_data = fake_variable_tensor.flatten()
    np.random.shuffle(flattened_data)
    fake_variable_tensor = flattened_data.reshape(shape)

    return fake_variable_tensor


def _get_input_dataset(input_data_dir, input_names):
    # inputs_name are a and b
    # [a.npy,b.npy]、[a_1.npy,b_1.npy] 、、、[a_n.npy,b_n.npy]
    dataset = None
    if input_data_dir is not None:
        dats = dict()
        dataset = list()
        for name in os.listdir(input_data_dir):
            if name.endswith(".npy"):
                index = 0
                split_name = splitext(name)[0]
                if not split_name in input_names:
                    parts = split_name.split("_")
                    if len(parts) > 1 and parts[-1].isnumeric():
                        index = int(parts[-1])
                        split_name = split_name[:-2]
                if not index in dats:
                    dats[index] = dict()
                dats[index][split_name] = np.load(os.path.join(input_data_dir, name))
            else:
                assert 0, f"File {name} can't be recognized, it should end with .npy"
        indexes = list(dats.keys())
        indexes.sort()
        for index in indexes:
            dataset.append(dats[index])
    return dataset


def gen_dataset(
    mod: tvm.IRModule,
    input_path: str = None,
    hist: Tuple = None,
    input_dataset: list = None,
) -> Tuple[List, Dict]:
    inputs_dtype, inputs_name, inputs_shapes = get_params_info(
        mod,
    )
    # A dummy sample for each input
    dataset = list()
    sample = dict()

    if input_dataset is not None:
        dataset = input_dataset
        sample = input_dataset[0]
    else:
        for i, var_name in enumerate(inputs_name):
            if input_path is not None:
                if input_path.endswith(".txt"):
                    sample[var_name] = np.reshape(
                        np.loadtxt(input_path).astype("float32"), inputs_shapes[i]
                    )
                else:
                    sample[var_name] = np.load(input_path)
                assert len(inputs_name) == 1, "Doesn't support specifying multiple outputs"
            elif hist is not None:
                sample[var_name] = gen_fake_variable_tensor(hist, inputs_shapes[i]).astype(
                    inputs_dtype[i]
                )
                assert len(inputs_name) == 1, "Doesn't support specifying multiple outputs"
            else:
                sample[var_name] = np.random.uniform(-1, 1, inputs_shapes[i]).astype(
                    inputs_dtype[i]
                )
        dataset.append(sample)

    for i, dat in enumerate(dataset):
        for name in inputs_name:
            assert name in dat, "Error: {} must be in the {}th input_dict keys({})!".format(
                name, i, dat.keys()
            )

    return dataset, sample


def gen_random_dataset(mod: tvm.IRModule, data_len=1) -> List:
    inputs_dtype, inputs_name, inputs_shapes = get_params_info(
        mod,
    )
    dataset = []
    for k in range(data_len):
        sample = {}
        # A dummy sample for each input
        for i, var_name in enumerate(inputs_name):
            sample[var_name] = np.random.uniform(-1, 1, inputs_shapes[i]).astype(inputs_dtype[i])
        dataset.append(sample)
    return dataset


def load_plugin(plugin_path):
    # Load plugin
    plugin_dir = dirname(plugin_path)

    current_dir = getcwd()
    # importlib.import_module can't support . in path, so we go to the location of plugin
    chdir(plugin_dir)

    plugin_module_name = splitext(basename(plugin_path))[0]
    module = importlib.import_module(plugin_module_name)

    # go back
    chdir(current_dir)
    return module


# (TODO) Remove this after upstream pattern
def parse_shape_string(inputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]" that
        indicates the desired shape for specific model inputs.

    Returns
    -------
    shape_dict: dict
        A dictionary mapping input names to their shape for use in relay frontend converters.
    """

    # Create a regex pattern that extracts each separate input mapping.
    # We want to be able to handle:
    # * Spaces inside arrays
    # * forward slashes inside names (but not at the beginning or end)
    # * colons inside names (but not at the beginning or end)
    pattern = r"(?:\w*)?[:\w.\/]+\:\s*\[\-?\d+(?:\,\s*\-?\d+)*\]"
    input_mappings = re.findall(pattern, inputs_string)
    if not input_mappings:
        raise argparse.ArgumentTypeError(
            "--input-shapes argument must be of the form "
            '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]"'
        )
    shape_dict = {}
    for mapping in input_mappings:
        # Remove whitespace.
        mapping = mapping.replace(" ", "")
        # Split mapping into name and shape.
        name, shape_string = mapping.rsplit(":", 1)
        # Convert shape string into a list of integers or Anys if negative.
        shape = [int(x) if int(x) > 0 else relay.Any() for x in shape_string.strip("][").split(",")]
        # Add parsed mapping to shape dictionary.
        shape_dict[name] = shape

    return shape_dict


def get_cu_hc_counter_map(model_path):
    cmd = "python -m dl verify " + model_path
    env = {
        **environ,
        "USE_HC_COUNTER": "1",
    }
    proc = subprocess.Popen(
        cmd,
        bufsize=0,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        encoding="utf-8",
    )
    map = {}
    find_kernel = False
    kernel_name = None
    while True:
        data = proc.stdout.readline()
        result = proc.poll()
        # process is done if result is valid
        if result is not None:
            return map

        if find_kernel is True:
            cu_time = int(data.split()[10])
            map[kernel_name].append(cu_time)
            find_kernel = False

        if "Cu {Kernel Name:" in data:
            kernel_name = data.split()[6].split("}")[0]
            map[kernel_name] = []
            find_kernel = True


def gen_inputs(mod):
    inputs_dtype, inputs_name, inputs_shape = get_params_info(mod)

    params = dict()

    for i, input_name in enumerate(inputs_name):
        input_shape = inputs_shape[i]
        input_dtype = inputs_dtype[i]
        # Random input
        np_data = np.random.uniform(-5, 5, size=input_shape).astype(input_dtype)
        params[input_name] = tvm.nd.array(np_data)

    return params


def load_dleol():
    from dl.support import using_dlcu

    if using_dlcu():
        # DL plugins
        try:
            import plugin_register
        except Exception as e:
            sys.stderr.write(f"Error happens when importing plugin_register: {repr(e)}\n")

        try:
            import dleol
        except Exception as e:
            sys.stderr.write(f"Error happens when importing dleol: {repr(e)}\n")


def load_custom_plugin(library, module):
    # Customer plugins
    if library is not None:
        from dl.op import load_op_library

        for l in library:
            load_op_library(l)

    if module is not None:
        for m in module:
            module_path, module_name = split(m)
            spec = importlib.util.spec_from_file_location(module_name, m)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)


def parse_name_string(outputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "output0 output1" that
        indicates the model outputs.

    Returns
    -------
    outputs: list
        A list of output names
    """
    return outputs_string.split()


def add_plugin_argument(parser):
    parser.add_argument(
        "--library",
        help='Path of libraries which implement custom ops, format is "x.so y.so"',
        type=parse_name_string,
        default=None,
    )
    parser.add_argument(
        "--module",
        help='Path of python modules which implement custom ops, format is "x.py y.py"',
        type=parse_name_string,
        default=None,
    )


def check_unsupported_ops(mod, target):
    if target == "llvm":
        op_freqs = relay.analysis.list_op_freqs(mod)
        unsupported_ops = []
        if "dl.softmax" in op_freqs.keys():
            unsupported_ops.append("dl.softmax")

        if len(unsupported_ops):
            raise TVMCException(
                f"This model has some ops which doesn't support {target}: {unsupported_ops}"
            )

def compare_consine(data1,data2):
    cos_similarity = 1 - spatial.distance.cosine(data1, data2)
        # logger.info("Batch idx: {}, Compare output name: {} and {}, cos similarity: {}"
        #                 .format(idx, cpu_output_name, gpu_output_name, cos_similarity))
    print(f"cos_similarity:{cos_similarity}")
    if cos_similarity < 0.999 or np.isnan(cos_similarity):
        diff_abs_val = np.abs(np.array(data1) - np.array(data2))
        max_idx = np.argmax(diff_abs_val)
        # logging.error("Max Error = {}".format(np.max(diff_abs_val)))
        # logging.error("idx: {}, cpu: {}, gpu: {}"
        #     .format(max_idx, cpu_output_data_flatten[max_idx], gpu_output_data_flatten[max_idx]))
        print(f"data1:{data1} vs \ndata2:{data2}")
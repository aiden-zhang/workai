from os import lseek
from pydlnne_modulator import *
import dlnne as nne
import pydlnne
import pydlcuda
import pycuda.driver as cuda
import numpy as np
import time
import pickle
import copy
import os
import hashlib
import configparser
from dl import op



BasePath=os.path.dirname(os.path.abspath(__file__))
plugin_opt_so=os.path.join(BasePath,"../../lib/libDLNms_opt_plugin.so")

op.load_op_library(plugin_opt_so)

front_end = os.path.join(BasePath,"../../include/NMS/tvm_op/front_end.py")
tvm_opt_so = os.path.join(BasePath,"../../lib/libDLNms_opt_tvm.so")

print('front_end:', front_end)

def time2ms(time_stamp):
    return int(round(time_stamp * 1000))

class Binding:
    def __init__(self, mem, size, shape, dtype):
        self.mem = mem
        self.size = size
        self.shape = shape
        self.dtype = dtype

def get_dtype(type):
    if type == pydlnne.DataType.kFLOAT32:
        dtype = np.float32
    elif type == pydlnne.DataType.kFLOAT16:
        dtype = np.float16
    elif type == pydlnne.DataType.kUINT8:
        dtype = np.uint8
    elif type == pydlnne.DataType.kUINT16:
        dtype = np.uint16
    elif type == pydlnne.DataType.kUINT32:
        dtype = np.uint32
    elif type == pydlnne.DataType.kINT8:
        dtype = np.int8
    elif type == pydlnne.DataType.kINT16:
        dtype = np.int16
    elif type == pydlnne.DataType.kINT32:
        dtype = np.int32
    elif type == pydlnne.DataType.kINT64:
        dtype = np.int64
    elif type == pydlnne.DataType.kBOOL:
        dtype = np.bool
    else:
        raise Exception("Unknown data type: " + str(type))
    return dtype

weight_share_configs = {
        "0": {
            "weight_mode": nne.WeightShareMode.single,
            "cluster_cfg": nne.ClusterConfig.cluser0,
        },
        "1": {
            "weight_mode": nne.WeightShareMode.single,
            "cluster_cfg": nne.ClusterConfig.cluser1,
        },
        "2": {
            "weight_mode": nne.WeightShareMode.single,
            "cluster_cfg": nne.ClusterConfig.cluser2,
        },
        "3": {
            "weight_mode": nne.WeightShareMode.single,
            "cluster_cfg": nne.ClusterConfig.cluser3,
        },
        "01": {
            "weight_mode": nne.WeightShareMode.share2,
            "cluster_cfg": nne.ClusterConfig.cluser01,
        },
        "23": {
            "weight_mode": nne.WeightShareMode.share2,
            "cluster_cfg": nne.ClusterConfig.cluser23,
        },
        "0123": {
            "weight_mode": nne.WeightShareMode.share4,
            "cluster_cfg": nne.ClusterConfig.cluser0123,
        },
    }

class DlnneEngine():
    def __init__(   self, model_path, inputs_shape_dict, output_op_list, 
                    max_batch_size, seq_len, config_key, engine_log=None):
        self.model_path = model_path
        self.engine = None
        self.context = None
        self.bindings = {}
        self.output_op_list = output_op_list
        self.model_inputs_shape_dict = inputs_shape_dict
        self.engine_input_shape = None
        self.hashcode = self.md5hash(model_path)
        self.infer_time = 0
        self.infer_counter = 0
        self.engine_log = engine_log
        self._create_engine_context(max_batch_size, seq_len, config_key)
    
    def __del__(self):
        if len(self.bindings) > 0:
            for binding_name in self.bindings:
                if isinstance(self.bindings[binding_name], Binding):
                    pydlcuda.DeviceMemory.free(self.bindings[binding_name].mem)
    
    def md5hash(self, file_path):
        import hashlib
        md5_value = hashlib.md5()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(2048)
                if not data:
                    break
                md5_value.update(data)
        return md5_value.hexdigest()
    
    def _create_engine_context(self, max_batch_size = 1, seq_len = 0, config_key="0"):
        weight_mode = weight_share_configs[config_key]["weight_mode"]
        cluster_cfg = weight_share_configs[config_key]["cluster_cfg"]
        print("weight_mode: %s, cluster_cfg: %s" % (weight_mode, cluster_cfg))

        serilized_model_name = "py_" + str(self.hashcode) + "_" + str(max_batch_size) + "_" + str(seq_len) + ".engine"
        serilized_model_path = os.path.join("./DUMP", serilized_model_name)

        if os.path.isfile(serilized_model_path):
            with open(serilized_model_path, "rb") as f:
                self.engine = nne.deserialize(f.read())
                self.context = self.engine.create_execution_context(cluster_cfg)
        else:
            with nne.Builder() as builder, nne.Parser() as parser:     
                network = builder.create_network()
                builder.config.ws_mode = weight_mode
                builder.config.max_batch_size = max_batch_size

                [parser.register_output(key) for key in self.output_op_list]
                [parser.register_input(key, self.model_inputs_shape_dict[key]) for key, value in self.model_inputs_shape_dict.items()]
                parser.register_user_op(tvm_opt_so, front_end, 'custom_op')
                parser.parse(self.model_path, network)
                self.engine = builder.build_engine(network)

                # if not os.path.exists("./DUMP"):
                #     os.mkdir("./DUMP")
                # self.dump_engine(serilized_model_path)
                self.context = self.engine.create_execution_context(cluster_cfg)
        
        # mem alloc
        for index in range(self.engine.num_bindings):
            # import pdb 
            # pdb.set_trace()
            binding_name = self.engine.get_binding_name(index)
            shape = self.engine.get_binding_shape(index)
            dtype = get_dtype(self.engine.get_binding_dtype(index))
            vol = max_batch_size

            for s in shape:
                vol *= s
            if dtype == np.bool:
                size = vol
            else:
                size = vol * dtype(1).nbytes
            mem = pydlcuda.mem_alloc(size)
            self.bindings[binding_name] = Binding(mem, size, shape, dtype)

        return self.engine, self.context
    
    def dump_engine(self, serilized_model_path):
        # use this function when first use DlnneBertEngine class
        assert self.engine is not None
        print("---Serialize \n")
        with open(serilized_model_path, "wb") as f:
            f.write(self.engine.serialize())
        print("---Serialize file save to ", serilized_model_path + "\n")

    
    def inference(self, input_data, batch_size):
        for key, value in input_data.items():
            cuda.memcpy_htod(self.bindings[key].mem, value)

        # exe_time = time2ms(time.time())
        t1 = time.time()
        self.context.execute(batch_size, [binding.mem.as_buffer(binding.size) for binding in self.bindings.values()])
        self.infer_counter += batch_size
        self.infer_time += time.time() - t1
        if self.engine_log:
            self.engine_log.info("process %d samples take %d ms, %.2f samples/sec" % \
                (self.infer_counter, self.infer_time*1000, self.infer_counter/self.infer_time))
        # print("execute time: ", time2ms(time.time()) - exe_time)

        output_batch_data = []
        for index in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(index):
                binding_name = self.engine.get_binding_name(index)
                output_shape = list(self.bindings[binding_name].shape)
                output_shape[0] *= batch_size
                infer_result = cuda.from_device(self.bindings[binding_name].mem,
                                                output_shape,
                                                self.bindings[binding_name].dtype)
                output_batch_data.append(infer_result)

        return output_batch_data

    def get_engine_input_shape(self):
        if self.engine is None:
            return 0
        for index in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(index)
            if self.engine.binding_is_input(index):
                return shape
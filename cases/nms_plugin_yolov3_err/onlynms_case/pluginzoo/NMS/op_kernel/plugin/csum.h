#pragma once

#include "utils.h"
#include "kernel.h"

using namespace dl::nne;
using namespace std;
using namespace dlNMS;

class CsumPlugin : public PluginExt {
 public:
  CsumPlugin(int axis, int only_1dim) {
    DebugInfo("construct CsumPlugin");

    axis_ = axis;
    only_1dim_ = only_1dim;
  }

  CsumPlugin(const void *data, size_t length) {
    DebugInfo("construct CsumPlugin sue Serialize data");

    const char *bufdata = reinterpret_cast<char const *>(data);
    size_t size = 0, offset = 0;

    DebugInfo(offset);
    size = sizeof(axis_);
    memcpy(&axis_, bufdata + offset, size);

    offset = size;

    size = sizeof(only_1dim_);
    memcpy(&only_1dim_, bufdata + offset, size);
    offset += size;

    size = sizeof(keep_data_len_);
    memcpy(&keep_data_len_, bufdata + offset, size);
    offset += size;

    keep_data_.resize(keep_data_len_);
    size = sizeof(int) * keep_data_len_;
    memcpy(keep_data_.data(), bufdata + offset, size);
    offset += size;

    size_t str_size = 0;
    size = sizeof(str_size);
    memcpy(&str_size, bufdata + offset, size);
    offset += size;
    size = sizeof(char) * (str_size);

    str_kernel_.assign(bufdata + offset, size);
    offset += size;

    DebugInfo("deserialize offset", offset);
    ser_data_ = 1;
  }

  ~CsumPlugin() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");
    return 1;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    DebugInfo("call GetOutputDimensions");

    assert(nbInputDims == 1);

    std::cout << std::endl;

    Dims data_dims;

    if (0 == index) {
      if (only_1dim_ != 0) {
        int size = inputs[0].d[0];
        for (size_t i = 1; i < inputs[0].nbDims; i++)
          size = size * inputs[0].d[i];

        data_dims.nbDims = 1;
        data_dims.d[0] = size;
      } else {
        data_dims.nbDims = inputs[0].nbDims;
        for (size_t i = 0; i < inputs[0].nbDims; i++)
          data_dims.d[i] = inputs[0].d[i];
      }

      return data_dims;
    }

    return data_dims;
  }

  bool SupportsFormat(const Dims *inputDims, int nbInputs,
                      const Dims *outputDims, int nbOutputs,
                      const DataType *inputTypes, const DataType *outputTypes,
                      Format format) const override {
    DebugInfo("call SupportsFormat");

    return true;
  }

  size_t GetWorkspaceSize(int maxBatchSize) const override {
    DebugInfo("call GetWorkspaceSize");

    return 0;
  }

  int Enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, void *stream) override {
    DebugInfo("call Enqueue");
    // only for debug
    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());

    if (err != CUDA_SUCCESS) {
      DebugInfo("err: load module failed...");
      return -1;
    }
    assert(err == CUDA_SUCCESS);
    const char *func_name = "fused_dl_csum_kernel0";
    err = cuModuleGetFunction(&func_get_, module, func_name);
    assert(err == CUDA_SUCCESS);

    CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);

    CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);

    /*
    keep_data_ = {batch, num_ele, batch_s, num_ele_s, o_s1}
    */
    int batch = keep_data_[0];
    int num_ele = keep_data_[1];
    int batch_s = keep_data_[2];
    int num_ele_s = keep_data_[3];
    int o_s1 = keep_data_[4];
    int o_s0 = num_ele;

    DebugInfo("inputs:", batch, num_ele, "\tstrides:", batch_s, num_ele_s);

    DebugInfo("outputs:", batch * num_ele, "\tstrides:", o_s0, o_s1);

    void *args[] = {&input0,  &batch, &num_ele, &batch_s, &num_ele_s,
                    &output0, &batch, &num_ele, &o_s0,    &o_s1};

    CUDA_KERNEL_NODE_PARAMS params = {
        func_get_, (uint32_t)batch, 1, 1, 256, 1, 1, 0, args, nullptr};
    cuLaunchKernel(func_get_, batch, 1, 1, 256, 1, 1, 0, nullptr, args,
                   nullptr);
    return 0;
  }

  void *GetGraph(int batchSize, const void *const *inputs, void **outputs,
                 void *workspace, void *stream) override {
    DebugInfo("call GetGraph");

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());

    if (err != CUDA_SUCCESS) {
      DebugInfo("err: load module failed...");
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);
    const char *func_name = "fused_dl_csum_kernel0";
    err = cuModuleGetFunction(&func_get_, module, func_name);
    assert(err == CUDA_SUCCESS);

    CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);

    CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);

    /*
    keep_data_ = {batch, num_ele, batch_s, num_ele_s, o_s1}
    */
    int batch = keep_data_[0];
    int num_ele = keep_data_[1];
    int batch_s = keep_data_[2];
    int num_ele_s = keep_data_[3];
    int o_s1 = keep_data_[4];
    int o_s0 = num_ele;

    DebugInfo("inputs:", batch, num_ele, "\tstrides:", batch_s, num_ele_s);

    DebugInfo("outputs:", batch * num_ele, "\tstrides:", o_s0, o_s1);

    void *args[] = {&input0,  &batch, &num_ele, &batch_s, &num_ele_s,
                    &output0, &batch, &num_ele, &o_s0,    &o_s1};

    cuGraphCreate(&graph_, 0);
    CUDA_KERNEL_NODE_PARAMS params = {
        func_get_, (uint32_t)batch, 1, 1, 256, 1, 1, 0, args, nullptr};
    cuGraphAddKernelNode(&kernel_node_, graph_, nullptr, 0, &params);

    return reinterpret_cast<void *>(graph_);
  }

  void ConfigurePlugin(PluginTensorDesc const *in, int nbInput,
                       PluginTensorDesc const *out, int nbOutput,
                       int maxBatchSize) override {
    DebugInfo("Call Csum ConfigurePlugin");

    num_inputs_ = nbInput;
    num_outputs_ = nbOutput;
    max_batch_size_ = maxBatchSize;
    inputs_strides_.resize(nbInput);
    inputs_dims_.resize(nbInput);

    for (int i = 0; i < nbInput; i++) {
      inputs_strides_[i].nbDims = in[i].dims.nbDims;
      for (int j = 0; j < in[i].dims.nbDims; j++) {
        uint64_t *ptr_strides = in[i].strides;
        inputs_strides_[i].d[j] = *(ptr_strides + j);
        inputs_dims_[i].d[j] = in[i].dims.d[j];
      }
    }

    output_strides_.resize(nbOutput);
    outputs_dims_.resize(nbOutput);
    for (int i = 0; i < nbOutput; i++) {
      output_strides_[i].nbDims = out[i].dims.nbDims;
      for (int j = 0; j < out[i].dims.nbDims; j++) {
        uint64_t *ptr_strides = out[i].strides;
        output_strides_[i].d[j] = *(ptr_strides + j);
        outputs_dims_[i].d[j] = out[i].dims.d[j];
      }
    }
  }

  const char *GetPluginType() const override {
    DebugInfo("call GetPluginType");
    return "custom_csum";
  }

  const char *GetPluginVersion() const override { return "1"; }

  void Destroy() override {
    if (kernel_node_) cuGraphDestroyNode(kernel_node_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    auto *plugin = new CsumPlugin(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    name_space_ = pluginNamespace;
  }

  const char *GetPluginNamespace() const override { return name_space_.data(); }

  int Initialize() override {
    int batch = inputs_dims_[0].d[0];
    int num_ele = inputs_dims_[0].d[1];
    int batch_s = inputs_strides_[0].d[0];
    int num_ele_s = inputs_strides_[0].d[1];
    int o_s1 = output_strides_[0].d[0];
    keep_data_ = {batch, num_ele, batch_s, num_ele_s, o_s1};
    keep_data_len_ = keep_data_.size();

    // std::string plugin_kernel_path = getenv("DLNMS_PLUGIN_KERNEL_PATH");

    // std::string source_file_name_ = plugin_kernel_path + "/csum.cu";

    // std::string source_file_name = "./csum.cu";

    // ofstream out(source_file_name_,ios::app);
    // out<<cusm_src_file<<endl;
    // out.close();

    std::unique_ptr<CacheDirMgr> cache_dir_mgr_ =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_->Init();
    std::string file_name;
    bool gen_filename = cache_dir_mgr_->GenTmpFileName(file_name, "fb");
    DebugInfo("Generate file name state: ", gen_filename);
    assert(gen_filename);

    std::string source_file_name_;
    bool gen_cu = cache_dir_mgr_->GenTmpFileName(source_file_name_, "cu");

    ofstream out(source_file_name_,ios::app);
    out<<cusm_src_file<<endl;
    out.close();   

    std::string cmd =
        "dlcc -std=c++14 --cuda-gpu-arch=dlgpuc64 --cuda-device-only -w " +
        source_file_name_ + " -o " + file_name;

    DebugInfo(cmd);
    int cmd_flag = PluginCallSystem(cmd.c_str());
    assert(cmd_flag != 0);

    // serialize bc file to string
    if (SaveBcToString(file_name, str_kernel_)) {
      DebugInfo("SaveBcToString");
    } else {
      assert(false);
    }

    return 0;
  }

  void Terminate() override {}

  void Serialize(void *buffer) const override {
    DebugInfo("call CsumPlugin::Serialize");
    char *bufdata = reinterpret_cast<char *>(buffer);
    size_t size = 0, offset = 0;
    size = sizeof(axis_);
    memcpy(bufdata, &axis_, size);
    offset += size;

    size = sizeof(only_1dim_);
    memcpy(bufdata + offset, &only_1dim_, size);
    offset += size;

    size = sizeof(keep_data_len_);
    memcpy(bufdata + offset, &keep_data_len_, size);
    offset += size;

    size = sizeof(int) * keep_data_len_;
    memcpy(bufdata + offset, keep_data_.data(), size);
    offset += size;

    size_t str_len = str_kernel_.size();
    size = sizeof(str_len);
    memcpy(bufdata + offset, &str_len, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(char) * (str_len);
    memcpy(bufdata + offset, str_kernel_.c_str(), size);
    offset += size;
    DebugInfo(offset);
  }

  size_t GetSerializationSize() const override {
    DebugInfo("call CsumPlugin::GetSerializationSize");
    return (sizeof(axis_) + sizeof(only_1dim_) + sizeof(keep_data_len_) +
            sizeof(int) * keep_data_len_ + sizeof(size_t) +
            sizeof(char) * str_kernel_.size());
  }

 private:
  std::string name_space_;

  CUfunction func_get_{};
  int ser_data_{0};
  CUgraph graph_{nullptr};
  CUgraphNode kernel_node_{nullptr};
  std::vector<Dims> inputs_strides_;
  std::vector<Dims> output_strides_;
  std::vector<Dims> inputs_dims_;
  std::vector<Dims> outputs_dims_;
  int num_inputs_{0};
  int num_outputs_{0};
  int max_batch_size_{0};
  std::string str_kernel_;
  int keep_data_len_{0};
  std::vector<int> keep_data_;

  int axis_{-1};
  int only_1dim_{0};
};

class CsumPluginCreator : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call CsumPluginCreator::GetPluginName");
    return "custom_csum";  // spatial_transformer
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call CsumPluginCreator::GetFieldNames");
    mfc_.nbFields = 2;

    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 0) {
        const PluginField fields_ = {"axis", nullptr, kINT32, 1};  // int32
        v.push_back(fields_);
      }

      if (i == 1) {
        const PluginField fields_ = {"only_1dim", nullptr, kINT32, 1};  // int32
        v.push_back(fields_);
      }
    }

    mfc_.fields = v.data();
    return &mfc_;
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call CsumPluginCreator::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("CsumPluginCreator create plugin ");

    int axis;
    int only_1dim;

    axis = ((int *)mfc_.fields[0].data)[0];
    only_1dim = ((int *)mfc_.fields[1].data)[0];

    if ((std::string(name) == GetPluginName()) ||
        (std::string(name) == "dl.custom_csum") ||
        (std::string(name) == "csum")) {
      auto plugin = new CsumPlugin(axis, only_1dim);

      DebugInfo("suceess create CsumPluginCreator");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call CsumPluginCreator::DeserializePlugin");
    if ((std::string(name) == GetPluginName()) ||
        (std::string(name) == "dl.custom_csum") ||
        (std::string(name) == "custom_csum") || (std::string(name) == "csum")) {
      auto plugin = new CsumPlugin(serialData, serialLength);
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call CsumPluginCreator::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call CsumPluginCreator::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

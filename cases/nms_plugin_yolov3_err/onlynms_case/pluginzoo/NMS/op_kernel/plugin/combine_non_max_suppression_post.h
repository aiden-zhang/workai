#pragma once

#include "utils.h"
#include "kernel.h"

using namespace dl::nne;
using namespace std;
using namespace dlNMS;

class CombineNonMaxSuppressionPlugin : public PluginExt {
 public:
  CombineNonMaxSuppressionPlugin(int max_total_size) {
    DebugInfo("construct CombineNonMaxSuppressionPlugin");
    this->max_total_size = max_total_size;
  }

  CombineNonMaxSuppressionPlugin(const void *data, size_t length) {
    DebugInfo("construct CombineNonMaxSuppressionPlugin sue Serialize data");

    const char *bufdata = reinterpret_cast<char const *>(data);
    size_t size = 0, offset = 0;

    DebugInfo(offset);
    size = sizeof(max_total_size);
    memcpy(&max_total_size, bufdata + offset, size);
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
  }

  ~CombineNonMaxSuppressionPlugin() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");

    return 4;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    DebugInfo("call GetOutputDimensions");

    assert(nbInputDims == 5);

    Dims data_dims;

    if (0 == index) {
      data_dims.nbDims = 2;
      data_dims.d[0] = this->max_total_size;
      data_dims.d[1] = 2;
    }
    if (1 == index) {
      data_dims.nbDims = 2;
      data_dims.d[0] = this->max_total_size;
      data_dims.d[1] = 4;
    }
    if (2 == index) {
      data_dims.nbDims = 1;
      data_dims.d[0] = this->max_total_size;
    }
    if (3 == index) {
      data_dims.nbDims = 1;
      data_dims.d[0] = 1;
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
    DebugInfo(max_total_size);

    return 0;
  }

  int Enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, void *stream) override {
    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());

    DebugInfo("call cuModuleLoad");

    if (err != CUDA_SUCCESS) {
      DebugInfo("call cuModuleLoad failed....!");
      return -1;
    }
    assert(err == CUDA_SUCCESS);

    const char *func_name = "fused_dl_combine_non_max_suppression_post_kernel";
    err = cuModuleGetFunction(&func_get, module, func_name);
    assert(err == CUDA_SUCCESS);

    DebugInfo("call cuModuleGetFunction");

    CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input1 = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr input2 = reinterpret_cast<CUdeviceptr>(inputs[2]);
    CUdeviceptr input3 = reinterpret_cast<CUdeviceptr>(inputs[3]);
    CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(inputs[4]);

    CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
    CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
    CUdeviceptr output3 = reinterpret_cast<CUdeviceptr>(outputs[3]);
    /*
    keep_data_ = {
                  s0_s0, s0_s1, s0_s2, s0_ss0, s0_ss1, s0_ss2,
                  s1_s0, s1_s1, s1_s2, s1_ss0, s1_ss1, s1_ss2,
                  s2_s0, s2_s1, s2_s2, s2_ss0, s2_ss1, s2_ss2,
                  s4_s0, s4_s1, s4_ss0, s4_ss1,
                  o0_ss0, o0_ss1,
                  o1_ss0, o1_ss1,
                  o2_ss0
                }
    */
    int s0_s0 = keep_data_[0];
    int s0_s1 = keep_data_[1];
    int s0_s2 = keep_data_[2];
    int s0_ss0 = keep_data_[3];
    int s0_ss1 = keep_data_[4];
    int s0_ss2 = keep_data_[5];

    int s1_s0 = keep_data_[6];
    int s1_s1 = keep_data_[7];
    int s1_s2 = keep_data_[8];
    int s1_ss0 = keep_data_[9];
    int s1_ss1 = keep_data_[10];
    int s1_ss2 = keep_data_[11];

    int s2_s0 = keep_data_[12];
    int s2_s1 = keep_data_[13];
    int s2_s2 = keep_data_[14];
    int s2_ss0 = keep_data_[15];
    int s2_ss1 = keep_data_[16];
    int s2_ss2 = keep_data_[17];

    int s4_s0 = keep_data_[18];
    int s4_s1 = keep_data_[19];
    int s4_ss0 = keep_data_[20];
    int s4_ss1 = keep_data_[21];

    int o0_ss0 = keep_data_[22];
    int o0_ss1 = keep_data_[23];

    int o1_ss0 = keep_data_[24];
    int o1_ss1 = keep_data_[25];

    int o2_ss0 = keep_data_[26];

    int two = 2;
    int one = 1;
    int ouput1_s1 = o1_ss0 / 4;

    void *args[] = {&input0, &s0_s0, &s0_s1, &s0_s2, &s0_ss0, &s0_ss1, &s0_ss2,
                    &input1, &s1_s0, &s1_s1, &s1_s2, &s1_ss0, &s1_ss1, &s1_ss2,
                    &input2, &s2_s0, &s2_s1, &s2_s2, &s2_ss0, &s2_ss1, &s2_ss2,
                    //                    &input3,
                    &input4, &s4_s0, &s4_ss0,

                    &output0, &(this->max_total_size), &two, &o0_ss0, &o0_ss1,
                    &output1, &(this->max_total_size), &one, &ouput1_s1,
                    &o1_ss1, &output2, &(this->max_total_size), &o2_ss0,
                    &output3};

    DebugInfo(
        "====================================================================="
        "=");

    DebugInfo("input_shape_strides[0]:", s0_s0, s0_s1, s0_s2, "\t", s0_ss0,
              s0_ss1, s0_ss2);
    DebugInfo("input_shape_strides[1]:", s1_s0, s1_s1, s1_s2, "\t", s1_ss0,
              s1_ss1, s1_ss2);
    DebugInfo("input_shape_strides[2]:", s2_s0, s2_s1, s2_s2, "\t", s2_ss0,
              s2_ss1, s2_ss2);

    DebugInfo("input_shape_strides[4]:", s4_s0, "\t", s4_ss0);

    DebugInfo("out_shape_strides[0]:", this->max_total_size, two, "\t", o0_ss0,
              o0_ss1);
    DebugInfo("out_shape_strides[1]:", this->max_total_size, one, "\t",
              ouput1_s1, o1_ss1);
    DebugInfo("out_shape_strides[2]:", this->max_total_size, "\t", o2_ss0);

    CUDA_KERNEL_NODE_PARAMS params = {
        func_get, (uint32_t)((this->max_total_size + 255) / 256),
        1,        1,
        256,      1,
        1,        0,
        args,     nullptr};
    cuLaunchKernel(func_get, (uint32_t)((this->max_total_size + 255) / 256), 1,
                   1, 256, 1, 1, 0, nullptr, args, nullptr);

    return 0;
  }

  void *GetGraph(int batchSize, const void *const *inputs, void **outputs,
                 void *workspace, void *stream) override {
    DebugInfo("call GetGraph");

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());

    DebugInfo("call cuModuleLoad");

    if (err != CUDA_SUCCESS) {
      DebugInfo("call cuModuleLoad failed....!");
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);

    const char *func_name = "fused_dl_combine_non_max_suppression_post_kernel";
    err = cuModuleGetFunction(&func_get, module, func_name);
    assert(err == CUDA_SUCCESS);

    DebugInfo("call cuModuleGetFunction");

    CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input1 = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr input2 = reinterpret_cast<CUdeviceptr>(inputs[2]);
    CUdeviceptr input3 = reinterpret_cast<CUdeviceptr>(inputs[3]);
    CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(inputs[4]);

    CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
    CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
    CUdeviceptr output3 = reinterpret_cast<CUdeviceptr>(outputs[3]);
    /*
    keep_data_ = {
                  s0_s0, s0_s1, s0_s2, s0_ss0, s0_ss1, s0_ss2,
                  s1_s0, s1_s1, s1_s2, s1_ss0, s1_ss1, s1_ss2,
                  s2_s0, s2_s1, s2_s2, s2_ss0, s2_ss1, s2_ss2,
                  s4_s0, s4_s1, s4_ss0, s4_ss1,
                  o0_ss0, o0_ss1,
                  o1_ss0, o1_ss1,
                  o2_ss0
                }
    */
    int s0_s0 = keep_data_[0];
    int s0_s1 = keep_data_[1];
    int s0_s2 = keep_data_[2];
    int s0_ss0 = keep_data_[3];
    int s0_ss1 = keep_data_[4];
    int s0_ss2 = keep_data_[5];

    int s1_s0 = keep_data_[6];
    int s1_s1 = keep_data_[7];
    int s1_s2 = keep_data_[8];
    int s1_ss0 = keep_data_[9];
    int s1_ss1 = keep_data_[10];
    int s1_ss2 = keep_data_[11];

    int s2_s0 = keep_data_[12];
    int s2_s1 = keep_data_[13];
    int s2_s2 = keep_data_[14];
    int s2_ss0 = keep_data_[15];
    int s2_ss1 = keep_data_[16];
    int s2_ss2 = keep_data_[17];

    int s4_s0 = keep_data_[18];
    int s4_s1 = keep_data_[19];
    int s4_ss0 = keep_data_[20];
    int s4_ss1 = keep_data_[21];

    int o0_ss0 = keep_data_[22];
    int o0_ss1 = keep_data_[23];

    int o1_ss0 = keep_data_[24];
    int o1_ss1 = keep_data_[25];

    int o2_ss0 = keep_data_[26];

    int two = 2;
    int one = 1;
    int ouput1_s1 = o1_ss0 / 4;

    void *args[] = {&input0, &s0_s0, &s0_s1, &s0_s2, &s0_ss0, &s0_ss1, &s0_ss2,
                    &input1, &s1_s0, &s1_s1, &s1_s2, &s1_ss0, &s1_ss1, &s1_ss2,
                    &input2, &s2_s0, &s2_s1, &s2_s2, &s2_ss0, &s2_ss1, &s2_ss2,
                    //                    &input3,
                    &input4, &s4_s0, &s4_ss0,

                    &output0, &(this->max_total_size), &two, &o0_ss0, &o0_ss1,
                    &output1, &(this->max_total_size), &one, &ouput1_s1,
                    &o1_ss1, &output2, &(this->max_total_size), &o2_ss0,
                    &output3};

    DebugInfo(
        "====================================================================="
        "=");

    DebugInfo("input_shape_strides[0]:", s0_s0, s0_s1, s0_s2, "\t", s0_ss0,
              s0_ss1, s0_ss2);
    DebugInfo("input_shape_strides[1]:", s1_s0, s1_s1, s1_s2, "\t", s1_ss0,
              s1_ss1, s1_ss2);
    DebugInfo("input_shape_strides[2]:", s2_s0, s2_s1, s2_s2, "\t", s2_ss0,
              s2_ss1, s2_ss2);

    DebugInfo("input_shape_strides[4]:", s4_s0, "\t", s4_ss0);

    DebugInfo("out_shape_strides[0]:", this->max_total_size, two, "\t", o0_ss0,
              o0_ss1);
    DebugInfo("out_shape_strides[1]:", this->max_total_size, one, "\t",
              ouput1_s1, o1_ss1);
    DebugInfo("out_shape_strides[2]:", this->max_total_size, "\t", o2_ss0);

    CUDA_KERNEL_NODE_PARAMS params = {
        func_get, (uint32_t)((this->max_total_size + 255) / 256),
        1,        1,
        256,      1,
        1,        0,
        args,     nullptr};

    cuGraphCreate(&graph_, 0);
    err = cuGraphAddKernelNode(&kernelNode_, graph_, nullptr, 0, &params);

    assert(err == CUDA_SUCCESS);

    DebugInfo("call cuGraphAddKernelNode");

    return reinterpret_cast<void *>(graph_);
  }

  void ConfigurePlugin(PluginTensorDesc const *in, int nbInput,
                       PluginTensorDesc const *out, int nbOutput,
                       int maxBatchSize) override {
    DebugInfo("Call Filter_sort ConfigurePlugin");

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
    return "custom_combine_non_max_suppression_post";
  }

  const char *GetPluginVersion() const override { return "1"; }

  void Destroy() override {
    if (kernelNode_) cuGraphDestroyNode(kernelNode_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    auto *plugin = new CombineNonMaxSuppressionPlugin(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  const char *GetPluginNamespace() const override { return mNamespace.data(); }

  int Initialize() override {
    int s0_s0 = inputs_dims_[0].d[0];
    int s0_s1 = inputs_dims_[0].d[1];
    int s0_s2 = inputs_dims_[0].d[2];
    int s0_ss0 = inputs_strides_[0].d[0] / 4;
    int s0_ss1 = inputs_strides_[0].d[1] / 4;
    int s0_ss2 = inputs_strides_[0].d[2] / 4;

    int s1_s0 = inputs_dims_[1].d[0];
    int s1_s1 = inputs_dims_[1].d[1];
    int s1_s2 = inputs_dims_[1].d[2];
    int s1_ss0 = inputs_strides_[1].d[0];
    int s1_ss1 = inputs_strides_[1].d[1];
    int s1_ss2 = inputs_strides_[1].d[2];

    int s2_s0 = inputs_dims_[2].d[0];
    int s2_s1 = inputs_dims_[2].d[1];
    int s2_s2 = inputs_dims_[2].d[2];
    int s2_ss0 = inputs_strides_[2].d[0];
    int s2_ss1 = inputs_strides_[2].d[1];
    int s2_ss2 = inputs_strides_[2].d[2];

    int s4_s0 = inputs_dims_[4].d[0];
    int s4_s1 = inputs_dims_[4].d[1];
    int s4_ss0 = inputs_strides_[4].d[0];
    int s4_ss1 = inputs_strides_[4].d[1];

    int o0_ss0 = output_strides_[0].d[0];
    int o0_ss1 = output_strides_[0].d[1];

    int o1_ss0 = output_strides_[1].d[0];
    int o1_ss1 = output_strides_[1].d[1];

    int o2_ss0 = output_strides_[2].d[0];
    keep_data_ = {s0_s0,  s0_s1,  s0_s2,  s0_ss0, s0_ss1, s0_ss2, s1_s0,
                  s1_s1,  s1_s2,  s1_ss0, s1_ss1, s1_ss2, s2_s0,  s2_s1,
                  s2_s2,  s2_ss0, s2_ss1, s2_ss2, s4_s0,  s4_s1,  s4_ss0,
                  s4_ss1, o0_ss0, o0_ss1, o1_ss0, o1_ss1, o2_ss0};
    keep_data_len_ = keep_data_.size();

    // std::string plugin_kernel_path = getenv("DLNMS_PLUGIN_KERNEL_PATH");

    // std::string source_file_name_ =
    //     plugin_kernel_path + "/combine_non_max_suppression_post.cu";



    // std::string source_file_name_ = "./combine_non_max_suppression_post.cu";

    std::unique_ptr<CacheDirMgr> cache_dir_mgr_ =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_->Init();
    std::string file_name;
    bool gen_filename = cache_dir_mgr_->GenTmpFileName(file_name, "bc");
    DebugInfo("Generate file name state: ", gen_filename);
    assert(gen_filename);

    std::string source_file_name_;
    bool gen_cu = cache_dir_mgr_->GenTmpFileName(source_file_name_, "cu");

    ofstream out(source_file_name_,ios::app);
    out<<combine_src_file<<endl;
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
  size_t GetSerializationSize() const override {
    DebugInfo("call CombineNonMaxSuppressionPlugin::GetSerializationSize");
    return sizeof(this->max_total_size) + sizeof(keep_data_len_) +
           sizeof(int) * keep_data_len_ + sizeof(size_t) +
           sizeof(char) * str_kernel_.size();
  }
  void Terminate() override {}
  void Serialize(void *buffer) const override {
    DebugInfo("call CombineNonMaxSuppressionPlugin::Serialize");
    char *bufdata = reinterpret_cast<char *>(buffer);
    size_t offset = 0;
    size_t size = sizeof(this->max_total_size);
    memcpy(bufdata, &this->max_total_size, size);
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

 private:
  std::string mNamespace;

  CUfunction func_get{};
  int serData{0};
  CUgraph graph_{nullptr};
  CUgraphNode kernelNode_{nullptr};
  std::vector<Dims> inputs_strides_;
  std::vector<Dims> output_strides_;
  std::vector<Dims> inputs_dims_;
  std::vector<Dims> outputs_dims_;
  CUmodule module{nullptr};
  int num_inputs_{0};
  int num_outputs_{0};
  int max_batch_size_{0};
  std::vector<int> keep_data_;
  int keep_data_len_{0};

  int max_total_size{0};
  std::string str_kernel_;
};

class CombineNonMaxSuppressionPluginCreator : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call CombineNonMaxSuppressionPluginCreator::GetPluginName");
    return "custom_combine_non_max_suppression_post";  // spatial_transformer
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call CombineNonMaxSuppressionPluginCreator::GetFieldNames");
    mfc_.nbFields = 1;

    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 0) {
        const PluginField fields_ = {"max_total_size", nullptr, kINT32,
                                     1};  // int32
        v.push_back(fields_);
      }
    }

    mfc_.fields = v.data();
    return &mfc_;
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call CombineNonMaxSuppressionPluginCreator::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("CombineNonMaxSuppressionPluginCreator create plugin ");

    int max_total_size;

    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 0) {
        max_total_size = ((int *)mfc_.fields[0].data)[0];
      }
    }

    if ((std::string(name) == GetPluginName()) ||
        (std::string(name) == "dl.custom_combine_non_max_suppression_post") ||
        (std::string(name) == "combine_non_max_suppression_post") ||
        (std::string(name) == "custom_combine_non_max_suppression_post")) {
      auto plugin = new CombineNonMaxSuppressionPlugin(max_total_size);

      DebugInfo("suceess create SpatialTransformerPluginCreator");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call CombineNonMaxSuppressionPluginCreator::DeserializePlugin");
    if ((std::string(name) == GetPluginName()) ||
        (std::string(name) == "dl.custom_combine_non_max_suppression_post") ||
        (std::string(name) == "combine_non_max_suppression_post") ||
        (std::string(name) == "custom_combine_non_max_suppression_post")) {
      auto plugin =
          new CombineNonMaxSuppressionPlugin(serialData, serialLength);
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call CombineNonMaxSuppressionPluginCreator::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call CombineNonMaxSuppressionPluginCreator::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

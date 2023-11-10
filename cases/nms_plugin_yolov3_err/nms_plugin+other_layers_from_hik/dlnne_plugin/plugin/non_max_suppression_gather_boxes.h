#ifndef PLUGIN_SRC_YOLOV3_NON_MAX_SUPPRESSION_GATHER_BOXES_H__
#define PLUGIN_SRC_YOLOV3_NON_MAX_SUPPRESSION_GATHER_BOXES_H__
#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <vector>

#include "cuda.h"
#include "dlnne.h"
#include "utils.h"

using namespace dl::nne;
using namespace std;

class YoloV3NonMaxSuppressionGatherBoxPlugin : public PluginExt {
 public:
  YoloV3NonMaxSuppressionGatherBoxPlugin() {
    DebugInfo("construct YoloV3NonMaxSuppressionGatherBoxPlugin");

    m_inputDims.resize(3);
    ser_data_ = 1;
  }

  YoloV3NonMaxSuppressionGatherBoxPlugin(const void *data, size_t length) {
    DebugInfo(
        "construct YoloV3NonMaxSuppressionGatherBoxPlugin sue Serialize data");
    const char *bufdata = reinterpret_cast<char const *>(data);

    size_t size = 0, offset = 0;
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

    m_inputDims.resize(3);
    ser_data_ = 1;
  }

  ~YoloV3NonMaxSuppressionGatherBoxPlugin() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");
    return 1;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    DebugInfo("call GetOutputDimensions");

    assert(nbInputDims == 3);

    if (0 == index) {
      Dims data_dims;
      data_dims.nbDims = (inputs[1].nbDims - 1 + 2);
      for (int i = 0; i < (inputs[1].nbDims - 1); ++i) {
        data_dims.d[i] = inputs[1].d[i];
      }

      data_dims.d[(data_dims.nbDims - 2)] = inputs[0].d[(inputs[0].nbDims - 2)];
      data_dims.d[(data_dims.nbDims - 1)] = inputs[0].d[(inputs[0].nbDims - 1)];

      return data_dims;
    } else {
      assert(false);
    }
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
    int nbIn1_dims = keep_data_[0];
    int nbOut0_dims = keep_data_[1];
    int q = keep_data_[2];
    int B = keep_data_[3];
    int C = keep_data_[4];
    int N = keep_data_[5];
    int ids_len = keep_data_[6];
    int block_num = keep_data_[7];
    DebugInfo(nbIn1_dims, nbOut0_dims, q, B, C, N, ids_len, block_num);
    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());
    if (err != CUDA_SUCCESS) {
      std::cout << "err: " << err << ", load module failed..." << std::endl;
      return -1;
    }
    assert(err == CUDA_SUCCESS);

    const char *func_name = "gather_boxes_device_func_global";
    err = cuModuleGetFunction(&func_get_, module, func_name);

    if (err != CUDA_SUCCESS) {
      DebugInfo("err: cuModuleGetFunction failed...");
      return -1;
    }

    assert(err == CUDA_SUCCESS);

    CUdeviceptr input_boxes = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input_sorted_idx = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr input_sort_size = reinterpret_cast<CUdeviceptr>(inputs[2]);

    CUdeviceptr output_data = reinterpret_cast<CUdeviceptr>(outputs[0]);

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    const unsigned int block_x = 512;
    const unsigned int block_y = 1;
    const unsigned int block_z = 1;

    const unsigned int grid_x = block_num;
    const unsigned int grid_y = 1;
    const unsigned int grid_z = 1;

    void *args[] = {
        &input_boxes, &input_sorted_idx, &output_data,     &q, &B, &C,
        &N,           &ids_len,          &input_sort_size,
    };
    CUDA_KERNEL_NODE_PARAMS params = {func_get_, grid_x,  grid_y,  grid_z,
                                      block_x,   block_y, block_z, 0,
                                      args,      nullptr};
    cuLaunchKernel(func_get_, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                   0, nullptr, args, nullptr);

    return 0;
  }

  void *GetGraph(int batchSize, const void *const *inputs, void **outputs,
                 void *workspace, void *stream) override {
    DebugInfo("YoloV3NonMaxSuppressionGatherBoxPlugin call GetGraph");
    /*
    keep_data_ = {nbIn0_dims, nbOut0_dims, q, B, C, N, ids_len, block_num};
    */
    int nbIn1_dims = keep_data_[0];
    int nbOut0_dims = keep_data_[1];
    int q = keep_data_[2];
    int B = keep_data_[3];
    int C = keep_data_[4];
    int N = keep_data_[5];
    int ids_len = keep_data_[6];
    int block_num = keep_data_[7];
    DebugInfo(nbIn1_dims, nbOut0_dims, q, B, C, N, ids_len, block_num);
    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());
    if (err != CUDA_SUCCESS) {
      std::cout << "err: " << err << ", load module failed..." << std::endl;
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);

    const char *func_name = "gather_boxes_device_func_global";
    err = cuModuleGetFunction(&func_get_, module, func_name);

    if (err != CUDA_SUCCESS) {
      DebugInfo("err: cuModuleGetFunction failed...");
      return nullptr;
    }

    assert(err == CUDA_SUCCESS);

    CUdeviceptr input_boxes = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input_sorted_idx = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr input_sort_size = reinterpret_cast<CUdeviceptr>(inputs[2]);

    CUdeviceptr output_data = reinterpret_cast<CUdeviceptr>(outputs[0]);

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    const unsigned int block_x = 512;
    const unsigned int block_y = 1;
    const unsigned int block_z = 1;

    const unsigned int grid_x = block_num;
    const unsigned int grid_y = 1;
    const unsigned int grid_z = 1;

    cuGraphCreate(&graph_, 0);

    void *args[] = {
        &input_boxes, &input_sorted_idx, &output_data,     &q, &B, &C,
        &N,           &ids_len,          &input_sort_size,
    };
    CUDA_KERNEL_NODE_PARAMS params = {func_get_, grid_x,  grid_y,  grid_z,
                                      block_x,   block_y, block_z, 0,
                                      args,      nullptr};
    cuGraphAddKernelNode(&kernelNode_, graph_, nullptr, 0, &params);

    return reinterpret_cast<void *>(graph_);
  }

  void ConfigurePlugin(PluginTensorDesc const *in, int nbInput,
                       PluginTensorDesc const *out, int nbOutput,
                       int maxBatchSize) override {
    DebugInfo("Call Non_Max_Suppression_Gather_Boxes ConfigurePlugin");

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
        DebugInfo("input dims, strides: ", inputs_dims_[i].d[j],
                  inputs_strides_[i].d[j]);
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
        DebugInfo("output dims, strides: ", outputs_dims_[i].d[j],
                  output_strides_[i].d[j]);
      }
    }
  }

  const char *GetPluginType() const override {
    DebugInfo("call GetPluginType");
    return "custome_non_max_suppression_gather_boxes";
  }

  const char *GetPluginVersion() const override { return "1"; }

  void Destroy() override {
    if (kernelNode_) cuGraphDestroyNode(kernelNode_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    auto *plugin = new YoloV3NonMaxSuppressionGatherBoxPlugin(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    name_space_ = pluginNamespace;
  }

  const char *GetPluginNamespace() const override { return name_space_.data(); }

  int Initialize() override {
    DebugInfo("Call Initialize");
    int nbIn1_dims = inputs_strides_[1].nbDims;
    int nbOut0_dims = output_strides_[0].nbDims;
    int q = inputs_dims_[0].d[1];
    int B = outputs_dims_[0].d[0];
    int C = outputs_dims_[0].d[1];
    int N = outputs_dims_[0].d[nbOut0_dims - 2];
    int ids_len = inputs_dims_[1].d[nbIn1_dims - 1];

    int block_num = 1;
    for (int j = 0; j < (nbOut0_dims - 2); ++j) {
      block_num = block_num * outputs_dims_[0].d[j];
    }
    keep_data_ = {nbIn1_dims, nbOut0_dims, q, B, C, N, ids_len, block_num};
    keep_data_len_ = 8;
    DebugInfo(nbIn1_dims, nbOut0_dims, q, B, C, N, ids_len, block_num);

    std::string plugin_kernel_path;
    if (getenv("YOLOV3_PLUGIN_KERNEL_PATH") == nullptr) {
      DebugInfo(" error: not export YOLOV3_PLUGIN_KERNEL_PATH");
      assert(false);
    } else {
      plugin_kernel_path = std::string(getenv("YOLOV3_PLUGIN_KERNEL_PATH"));
    }

    std::string source_file_name_;
    std::unique_ptr<CacheDirMgr> cache_dir_mgr_0 =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_0->Init();
    bool gen_cu = cache_dir_mgr_0->GenTmpFileName(source_file_name_, "cu");
    const char *func_name_ = "gather_boxes_device_func_global_";
    const char *kernel_name = "gather_boxes_device_func_kernel";

    std::stringstream func_template;
    func_template << "\n#include \"" << plugin_kernel_path.c_str()
                  << "/non_max_suppression_gather_boxes_kernel.h\""
                  << std::endl;
    func_template
        << "extern \"C\" __global__ void " << func_name_
        << "(const float4* boxes_buffer,const int* ids_buffer,float4* "
           "out_boxes_buffer,const int* sort_size_buffer=nullptr){"
        << std::endl;
    func_template << kernel_name << "(boxes_buffer,ids_buffer,out_boxes_buffer,"
                  << q << "," << B << "," << C << "," << N << "," << ids_len
                  << ",sort_size_buffer);" << std::endl;
    func_template << "}" << std::endl;

    DebugInfo(func_template.str());
    std::ofstream ofile(source_file_name_, ios::out);
    ofile << func_template.str();
    ofile.close();

    std::unique_ptr<CacheDirMgr> cache_dir_mgr_ =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_->Init();
    std::string file_name;
    bool gen_filename = cache_dir_mgr_->GenTmpFileName(file_name, "bc");
    DebugInfo("Generate file name state: ", gen_filename);
    assert(gen_filename);

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
    DebugInfo(
        "call YoloV3NonMaxSuppressionGatherBoxPlugin::GetSerializationSize");
    return (sizeof(keep_data_len_) + sizeof(int) * keep_data_len_ +
            sizeof(size_t) + sizeof(char) * str_kernel_.size());
  }
  void Terminate() override {}
  void Serialize(void *buffer) const override {
    DebugInfo("call YoloV3NonMaxSuppressionGatherBoxPlugin::Serialize");
    char *bufdata = reinterpret_cast<char *>(buffer);
    size_t size = 0, offset = 0;
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
  std::string name_space_;
  CUfunction func_get_{};
  int ser_data_{0};
  CUgraph graph_{nullptr};
  CUgraphNode kernelNode_{nullptr};
  std::vector<Dims> m_inputDims;
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
};

class YoloV3NMSGBPluginCreator_ : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call YoloV3NMSGBPluginCreator_::GetPluginName");
    return "custome_non_max_suppression_gather_boxes";
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call YoloV3NMSGBPluginCreator_::GetFieldNames");
    // no attribute this op
    mfc_.nbFields = 0;
    return &mfc_;
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call YoloV3NMSGBPluginCreator_::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("YoloV3NMSGBPluginCreator_ create plugin ");

    if (std::string(name) == GetPluginName()) {
      auto plugin = new YoloV3NonMaxSuppressionGatherBoxPlugin();
      DebugInfo("suceess create YoloV3NMSGBPluginCreator_");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call YoloV3NMSGBPluginCreator_::DeserializePlugin");
    if (std::string(name) == GetPluginName()) {
      auto plugin =
          new YoloV3NonMaxSuppressionGatherBoxPlugin(serialData, serialLength);
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call YoloV3NMSGBPluginCreator_::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call YoloV3NMSGBPluginCreator_::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

#endif

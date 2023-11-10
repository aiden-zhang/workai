#ifndef PLUGIN_SRC_YOLOV3_NON_MAX_SUPPRESSION_H__
#define PLUGIN_SRC_YOLOV3_NON_MAX_SUPPRESSION_H__
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

class YoloV3NMSPlugin_ : public PluginExt {
 public:
  YoloV3NMSPlugin_(int center_point_box, int sorted) {
    DebugInfo("construct YoloV3NMSPlugin_");

    center_point_box_ = center_point_box;
    sorted_ = sorted;

    ser_data_ = 1;
  }

  YoloV3NMSPlugin_(const void *data, size_t length) {
    DebugInfo("construct YoloV3NMSPlugin_ sue Serialize data");

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

    ser_data_ = 1;
  }

  ~YoloV3NMSPlugin_() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");
    return 3;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    DebugInfo("call GetOutputDimensions");

    assert(nbInputDims == 6);

    if (0 == index) {
      Dims data_dims;
      data_dims.nbDims = 3;
      data_dims.d[0] = inputs[1].d[0];
      data_dims.d[1] = inputs[1].d[1];
      data_dims.d[2] = inputs[0].d[2];

      return data_dims;
    } else if (1 == index) {
      Dims data_dims;
      data_dims.nbDims = 2;
      data_dims.d[0] = inputs[1].d[0];
      data_dims.d[1] = inputs[1].d[1];
      return data_dims;
    } else if (2 == index) {
      Dims data_dims;
      data_dims.nbDims = 3;
      data_dims.d[0] = inputs[1].d[0];
      data_dims.d[1] = inputs[1].d[1];
      data_dims.d[2] = inputs[0].d[2];
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
    int block_num = keep_data_[0];
    int box_class_num = keep_data_[1];
    int B = keep_data_[2];
    int C = keep_data_[3];
    int box_s = keep_data_[4];
    int scores_s = keep_data_[5];

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());
    if (err != CUDA_SUCCESS) {
      DebugInfo("err:load module failed...");
      return -1;
    }
    assert(err == CUDA_SUCCESS);

    const char *func_name = "NonMaxSuppression_device_func_global";
    err = cuModuleGetFunction(&func_get_, module, func_name);

    if (err != CUDA_SUCCESS) {
      DebugInfo("err:  cuModuleGetFunction failed...");
      return -1;
    }

    assert(err == CUDA_SUCCESS);

    CUdeviceptr boxes_buffer = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr scores_buffer = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr max_output_boxes_per_class_buffer =
        reinterpret_cast<CUdeviceptr>(inputs[2]);
    CUdeviceptr iou_threshold_buffer = reinterpret_cast<CUdeviceptr>(inputs[3]);
    CUdeviceptr scores_threshold_buffer =
        reinterpret_cast<CUdeviceptr>(inputs[4]);
    CUdeviceptr sort_size_buffer = reinterpret_cast<CUdeviceptr>(inputs[5]);

    CUdeviceptr is_disable_buffer = reinterpret_cast<CUdeviceptr>(outputs[2]);
    CUdeviceptr boxIds_buffer = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUdeviceptr count_buffer = reinterpret_cast<CUdeviceptr>(outputs[1]);

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    const unsigned int block_x = 256;
    const unsigned int block_y = 1;
    const unsigned int block_z = 1;

    const unsigned int grid_x = block_num;
    const unsigned int grid_y = 1;
    const unsigned int grid_z = 1;

    void *args[] = {&boxes_buffer,
                    &scores_buffer,
                    &max_output_boxes_per_class_buffer,
                    &iou_threshold_buffer,
                    &scores_threshold_buffer,
                    &box_class_num,
                    &B,
                    &C,
                    &box_s,
                    &scores_s,
                    &is_disable_buffer,
                    &boxIds_buffer,
                    &count_buffer,
                    &sort_size_buffer};
    CUDA_KERNEL_NODE_PARAMS params = {func_get_, grid_x,  grid_y,  grid_z,
                                      block_x,   block_y, block_z, 0,
                                      args,      nullptr};
    cuLaunchKernel(func_get_, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                   0, nullptr, args, nullptr);

    return 0;
  }

  void *GetGraph(int batchSize, const void *const *inputs, void **outputs,
                 void *workspace, void *stream) override {
    DebugInfo("+YoloV3NMSPlugin_::GetGraph");
    /*
    keep_data_ = {block_num, box_class_num, B, C, box_s, scores_s};
    */
    int block_num = keep_data_[0];
    int box_class_num = keep_data_[1];
    int B = keep_data_[2];
    int C = keep_data_[3];
    int box_s = keep_data_[4];
    int scores_s = keep_data_[5];

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());
    if (err != CUDA_SUCCESS) {
      DebugInfo("err:load module failed...");
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);

    const char *func_name = "NonMaxSuppression_device_func_global";
    err = cuModuleGetFunction(&func_get_, module, func_name);

    if (err != CUDA_SUCCESS) {
      DebugInfo("err:  cuModuleGetFunction failed...");
      return nullptr;
    }

    assert(err == CUDA_SUCCESS);

    CUdeviceptr boxes_buffer = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr scores_buffer = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr max_output_boxes_per_class_buffer =
        reinterpret_cast<CUdeviceptr>(inputs[2]);
    CUdeviceptr iou_threshold_buffer = reinterpret_cast<CUdeviceptr>(inputs[3]);
    CUdeviceptr scores_threshold_buffer =
        reinterpret_cast<CUdeviceptr>(inputs[4]);
    CUdeviceptr sort_size_buffer = reinterpret_cast<CUdeviceptr>(inputs[5]);

    CUdeviceptr is_disable_buffer = reinterpret_cast<CUdeviceptr>(outputs[2]);
    CUdeviceptr boxIds_buffer = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUdeviceptr count_buffer = reinterpret_cast<CUdeviceptr>(outputs[1]);

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    const unsigned int block_x = 256;
    const unsigned int block_y = 1;
    const unsigned int block_z = 1;

    const unsigned int grid_x = block_num;
    const unsigned int grid_y = 1;
    const unsigned int grid_z = 1;

    cuGraphCreate(&graph_, 0);

    void *args[] = {&boxes_buffer,
                    &scores_buffer,
                    &max_output_boxes_per_class_buffer,
                    &iou_threshold_buffer,
                    &scores_threshold_buffer,
                    &box_class_num,
                    &B,
                    &C,
                    &box_s,
                    &scores_s,
                    &is_disable_buffer,
                    &boxIds_buffer,
                    &count_buffer,
                    &sort_size_buffer};
    CUDA_KERNEL_NODE_PARAMS params = {func_get_, grid_x,  grid_y,  grid_z,
                                      block_x,   block_y, block_z, 0,
                                      args,      nullptr};
    cuGraphAddKernelNode(&kernelNode_, graph_, nullptr, 0, &params);
    DebugInfo("-YoloV3NMSPlugin_::GetGraph");
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
    return "custom_non_max_suppression";
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call GetPluginVersion");
    return "1";
  }

  void Destroy() override {
    DebugInfo("call Destroy");
    if (kernelNode_) cuGraphDestroyNode(kernelNode_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    DebugInfo("call Clone");
    auto *plugin = new YoloV3NMSPlugin_(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call SetPluginNamespace");
    name_space_ = std::string(pluginNamespace);
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call YoloV3NMSPlugin_ GetPluginNamespace");
    return name_space_.data();
  }

  int Initialize() override {
    DebugInfo("Call Initialize");
    int nbIn1_dims = inputs_strides_[1].nbDims;
    int block_num = 1;
    for (int i = 0; i < (nbIn1_dims - 1); ++i) {
      block_num = block_num * inputs_dims_[1].d[i];
    }
    int box_class_num = inputs_dims_[0].d[1];
    int B = inputs_dims_[1].d[0];
    int C = inputs_dims_[1].d[1];
    int box_s = inputs_dims_[0].d[2];
    int scores_s = inputs_dims_[1].d[2];

    keep_data_ = {block_num, box_class_num, B, C, box_s, scores_s};
    keep_data_len_ = 6;

    std::string plugin_kernel_path;
    if (getenv("YOLOV3_PLUGIN_KERNEL_PATH") == nullptr) {
      DebugInfo(" error: not export YOLOV3_PLUGIN_KERNEL_PATH");
      return -1;
    } else {
      plugin_kernel_path = std::string(getenv("YOLOV3_PLUGIN_KERNEL_PATH"));
    }

    std::string source_file_name_ =
        plugin_kernel_path + "/non_max_suppression.cu";
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
    DebugInfo("call YoloV3NMSPlugin_::GetSerializationSize");
    return (sizeof(keep_data_len_) + sizeof(int) * keep_data_len_ +
            sizeof(size_t) + sizeof(char) * str_kernel_.size());
  }
  void Terminate() override { DebugInfo("call YoloV3NMSPlugin_::Terminate"); }
  void Serialize(void *buffer) const override {
    DebugInfo("call YoloV3NMSPlugin_::Serialize");
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

  int center_point_box_;
  int sorted_;
};

class YoloV3NMSPluginCreator_ : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call YoloV3NMSPluginCreator_::GetPluginName");
    return "custom_non_max_suppression";
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call YoloV3NMSPluginCreator_::GetFieldNames");

    mfc_.nbFields = 2;
    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 0) {
        const PluginField fields_ = {"center_point_box", nullptr, kINT32,
                                     1};  // int32
        v.push_back(fields_);
      } else if (i == 1) {
        const PluginField fields_ = {"sorted", nullptr, kINT32, 1};  //
        v.push_back(fields_);
      }
    }

    mfc_.fields = v.data();
    return &mfc_;
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call YoloV3NMSPluginCreator_::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("YoloV3NMSPluginCreator_ create plugin ");

    int center_point_box;
    int sorted;
    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 0) {
        center_point_box = ((int *)mfc_.fields[i].data)[0];
      } else if (i == 1) {
        sorted = ((int *)mfc_.fields[i].data)[0];
      }
    }

    if (std::string(name) == GetPluginName()) {
      auto plugin = new YoloV3NMSPlugin_(center_point_box, sorted);
      DebugInfo("suceess create YoloV3NMSPluginCreator_");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call YoloV3NMSPluginCreator_::DeserializePlugin");

    if (std::string(name) == GetPluginName()) {
      auto plugin = new YoloV3NMSPlugin_(serialData, serialLength);
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call YoloV3NMSPluginCreator_::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call YoloV3NMSPluginCreator_::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

#endif

#ifndef PLUGIN_SRC_YOLOV3_BOX_H___
#define PLUGIN_SRC_YOLOV3_BOX_H___
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

class YoloV3BoxPlugin_ : public PluginExt {
 public:
  YoloV3BoxPlugin_(int align_out, std::string data_format) {
    DebugInfo("construct YoloV3BoxPlugin_");

    align_out_ = align_out;
    data_format_ = data_format;

    class_num = 80;

    inputs_dims_.resize(4);
    ser_data_ = 1;
  }

  YoloV3BoxPlugin_(const void *data, size_t length) {
    DebugInfo("construct YoloV3BoxPlugin_ sue Serialize data");

    const char *bufdata = reinterpret_cast<char const *>(data);
    size_t size = 0, offset = 0;

    size = sizeof(keep_data_len_);
    memcpy(&keep_data_len_, bufdata + offset, size);
    offset += size;

    keep_data_.resize(keep_data_len_);
    size = sizeof(int) * keep_data_len_;
    memcpy(keep_data_.data(), bufdata + offset, size);
    offset += size;

    size = sizeof(is_normal_stride_);
    memcpy(&is_normal_stride_, bufdata + offset, size);
    offset += size;

    DebugInfo(offset);
    size = sizeof(align_out_);
    memcpy(&align_out_, bufdata + offset, size);
    offset += size;

    size_t str_size = 0;
    size = sizeof(str_size);
    memcpy(&str_size, bufdata + offset, size);
    offset += size;
    size = sizeof(char) * (str_size);

    str_kernel_.assign(bufdata + offset, size);
    offset += size;

    DebugInfo(offset);
    size = sizeof(class_num);
    memcpy(&class_num, bufdata + offset, size);
    offset += size;

    DebugInfo(offset);

    inputs_dims_.resize(4);
    ser_data_ = 1;
  }

  ~YoloV3BoxPlugin_() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");
    return 1;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    DebugInfo("call GetOutputDimensions");

    assert(nbInputDims == 4);
    // current just support nhwc

    if (0 == index) {
      if ((data_format_[0] == 'N') && (data_format_[1] == 'H') &&
          (data_format_[2] == 'W') && (data_format_[3] == 'C')) {
        // nhwc
        assert(inputs[0].nbDims == 4);
        assert(inputs[1].nbDims == 2);
        assert(inputs[2].nbDims == 1);
        assert(inputs[3].nbDims == 2);

        int out_N = inputs[0].d[0];
        int out_H = inputs[0].d[1];
        int out_W = inputs[0].d[2];
        int out_anchors = inputs[1].d[0];
        int out_coord = 4;  // output point num
        int in_C = inputs[0].d[3];

        Dims data_dims;
        data_dims.nbDims = 5;
        data_dims.d[0] = out_N;
        data_dims.d[1] = out_H;
        data_dims.d[2] = out_W;
        data_dims.d[3] = out_anchors;
        data_dims.d[4] = out_coord;

        return data_dims;
      } else {
        assert(false);
      }
    } else {
      assert(false);
    }
  }

  bool SupportsFormat(const Dims *inputDims, int nbInputs,
                      const Dims *outputDims, int nbOutputs,
                      const DataType *inputTypes, const DataType *outputTypes,
                      Format format) const override {
    DebugInfo("call SupportsFormat");
    if ((data_format_[0] == 'N') && (data_format_[1] == 'H') &&
        (data_format_[2] == 'W') && (data_format_[3] == 'C')) {
      return true;
    } else {
      std::cout << "not support format: " << data_format_ << std::endl;
      return false;
    }
  }

  size_t GetWorkspaceSize(int maxBatchSize) const override {
    DebugInfo("call GetWorkspaceSize");
    return 0;
  }

  int Enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, void *stream) override {
    DebugInfo("DLBOx call Enqueue");

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    int out_N = keep_data_[0];
    int out_H = keep_data_[1];
    int out_W = keep_data_[2];
    int out_anchors = keep_data_[3];
    int total_size0 = keep_data_[4];
    DebugInfo("keep_data_: ", out_N, out_H, out_W, out_anchors, total_size0);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());
    if (err != CUDA_SUCCESS) {
      DebugInfo("err: load module failed...");
      return -1;
    }
    assert(err == CUDA_SUCCESS);
    DebugInfo("is_normal_stride_: ", is_normal_stride_);
    if (is_normal_stride_) {
      const char *func_name = "dl_boxes_float4_device_func_global_";
      err = cuModuleGetFunction(&func_get_, module, func_name);

      if (err != CUDA_SUCCESS) {
        DebugInfo("err: cuModuleGetFunction failed...");
        return -1;
      }

      assert(err == CUDA_SUCCESS);
    } else {
      const char *func_name = "dl_boxes_float4_device_func_global_aligned_";
      err = cuModuleGetFunction(&func_get_, module, func_name);

      if (err != CUDA_SUCCESS) {
        DebugInfo("err: cuModuleGetFunction failed...");
        return -1;
      }

      assert(err == CUDA_SUCCESS);
    }

    CUdeviceptr input_feature = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input_prior_anchors = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr input_image_shape = reinterpret_cast<CUdeviceptr>(inputs[2]);
    CUdeviceptr real_image_shape = reinterpret_cast<CUdeviceptr>(inputs[3]);

    CUdeviceptr output_data = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    const unsigned int block_x = 128;
    const unsigned int block_y = 1;
    const unsigned int block_z = 1;

    const unsigned int grid_x = ((total_size0 + block_x - 1) / block_x);
    const unsigned int grid_y = 1;
    const unsigned int grid_z = 1;

    if (is_normal_stride_) {
      void *args[] = {
          &input_feature,
          &input_prior_anchors,
          &input_image_shape,
          &real_image_shape,
          &out_N,
          &out_H,
          &out_W,
          &out_anchors,
          &class_num,
          &output_data,
      };
      cuLaunchKernel(func_get_, grid_x, grid_y, grid_z, block_x, block_y,
                     block_z, 0, nullptr, args, nullptr);
    } else {
      void *args[] = {
          &input_feature,
          &input_prior_anchors,
          &input_image_shape,
          &real_image_shape,
          &out_N,
          &out_H,
          &out_W,
          &out_anchors,
          &class_num,
          &output_data,
      };
      cuLaunchKernel(func_get_, grid_x, grid_y, grid_z, block_x, block_y,
                     block_z, 0, nullptr, args, nullptr);
    }

    return 0;
  }

  void *GetGraph(int batchSize, const void *const *inputs, void **outputs,
                 void *workspace, void *stream) override {
    DebugInfo("DLBOx call GetGraph");

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    int out_N = keep_data_[0];
    int out_H = keep_data_[1];
    int out_W = keep_data_[2];
    int out_anchors = keep_data_[3];
    int total_size0 = keep_data_[4];
    DebugInfo("keep_data_: ", out_N, out_H, out_W, out_anchors, total_size0);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());
    if (err != CUDA_SUCCESS) {
      DebugInfo("err: load module failed...");
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);
    DebugInfo("is_normal_stride_: ", is_normal_stride_);
    if (is_normal_stride_) {
      const char *func_name = "dl_boxes_float4_device_func_global_";
      err = cuModuleGetFunction(&func_get_, module, func_name);

      if (err != CUDA_SUCCESS) {
        DebugInfo("err: cuModuleGetFunction failed...");
        return nullptr;
      }

      assert(err == CUDA_SUCCESS);
    } else {
      const char *func_name = "dl_boxes_float4_device_func_global_aligned_";
      err = cuModuleGetFunction(&func_get_, module, func_name);

      if (err != CUDA_SUCCESS) {
        DebugInfo("err: cuModuleGetFunction failed...");
        return nullptr;
      }

      assert(err == CUDA_SUCCESS);
    }

    CUdeviceptr input_feature = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input_prior_anchors = reinterpret_cast<CUdeviceptr>(inputs[1]);
    CUdeviceptr input_image_shape = reinterpret_cast<CUdeviceptr>(inputs[2]);
    CUdeviceptr real_image_shape = reinterpret_cast<CUdeviceptr>(inputs[3]);

    CUdeviceptr output_data = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    const unsigned int block_x = 128;
    const unsigned int block_y = 1;
    const unsigned int block_z = 1;

    const unsigned int grid_x = ((total_size0 + block_x - 1) / block_x);
    const unsigned int grid_y = 1;
    const unsigned int grid_z = 1;

    cuGraphCreate(&graph_, 0);

    if (is_normal_stride_) {
      void *args[] = {
          &input_feature,
          &input_prior_anchors,
          &input_image_shape,
          &real_image_shape,
          &out_N,
          &out_H,
          &out_W,
          &out_anchors,
          &class_num,
          &output_data,
      };
      CUDA_KERNEL_NODE_PARAMS params = {func_get_, grid_x,  grid_y,  grid_z,
                                        block_x,   block_y, block_z, 0,
                                        args,      nullptr};
      cuGraphAddKernelNode(&kernelNode_, graph_, nullptr, 0, &params);
    } else {
      void *args[] = {
          &input_feature,
          &input_prior_anchors,
          &input_image_shape,
          &real_image_shape,
          &out_N,
          &out_H,
          &out_W,
          &out_anchors,
          &class_num,
          &output_data,
      };
      CUDA_KERNEL_NODE_PARAMS params = {func_get_, grid_x,  grid_y,  grid_z,
                                        block_x,   block_y, block_z, 0,
                                        args,      nullptr};
      cuGraphAddKernelNode(&kernelNode_, graph_, nullptr, 0, &params);
    }

    return reinterpret_cast<void *>(graph_);
  }

  void ConfigurePlugin(PluginTensorDesc const *in, int nbInput,
                       PluginTensorDesc const *out, int nbOutput,
                       int maxBatchSize) override {
    DebugInfo("Call boxes ConfigurePlugin");

    assert(nbInputs == 4);
    assert(nbOutputs == 1);

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
    return "custom_boxes_plugin";
  }

  const char *GetPluginVersion() const override { return "1"; }

  void Destroy() override {
    if (kernelNode_) cuGraphDestroyNode(kernelNode_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    auto *plugin = new YoloV3BoxPlugin_(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    name_space_ = pluginNamespace;
  }

  const char *GetPluginNamespace() const override { return name_space_.data(); }

  int Initialize() override {
    int nbIn0_dims = inputs_strides_[0].nbDims;
    assert(nbIn0_dims == 4);
    int ss00 = inputs_strides_[0].d[0];
    int ss01 = inputs_strides_[0].d[1];
    int ss02 = inputs_strides_[0].d[2];
    int ss03 = inputs_strides_[0].d[3];
    int s00 = inputs_dims_[0].d[0];
    int s01 = inputs_dims_[0].d[1];
    int s02 = inputs_dims_[0].d[2];
    int s03 = inputs_dims_[0].d[3];

    int nbIn1_dims = inputs_strides_[1].nbDims;
    assert(nbIn1_dims == 2);
    int ss10 = inputs_strides_[1].d[0];
    int ss11 = inputs_strides_[1].d[1];
    int s10 = inputs_dims_[1].d[0];
    int s11 = inputs_dims_[1].d[1];

    int nbIn2_dims = inputs_strides_[2].nbDims;
    assert(nbIn2_dims == 1);
    int ss20 = inputs_strides_[2].d[0];
    int s20 = inputs_dims_[2].d[0];

    int nbIn3_dims = inputs_strides_[3].nbDims;
    assert(nbIn3_dims == 2);
    int ss30 = inputs_strides_[3].d[0];
    int ss31 = inputs_strides_[3].d[1];
    int s30 = inputs_dims_[3].d[0];
    int s31 = inputs_dims_[3].d[1];

    int out_coord = 4;  // output point num
    int total_size0 = s00 * s01 * s02 * s10 * out_coord;
    total_size0 = total_size0 / 4;  // float4

    keep_data_ = {s00, s01, s02, s10, total_size0};
    keep_data_len_ = 5;

    if ((ss03 == 1) && (ss02 == s03) && (ss01 == s03 * s02) &&
        (ss00 == s03 * s02 * s01) && (ss11 == 1) && (ss10 == s11) &&
        (ss20 == 1) && (ss31 == 1) && (ss30 == s31)) {
      is_normal_stride_ = true;
    } else {
      is_normal_stride_ = false;
    }

    std::string plugin_kernel_path;
    if (getenv("YOLOV3_PLUGIN_KERNEL_PATH") == nullptr) {
      DebugInfo("error: not export YOLOV3_PLUGIN_KERNEL_PATH");
      return -1;
    } else {
      plugin_kernel_path = std::string(getenv("YOLOV3_PLUGIN_KERNEL_PATH"));
    }

    std::string source_file_name_;
    std::unique_ptr<CacheDirMgr> cache_dir_mgr_0 =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_0->Init();
    bool gen_cu = cache_dir_mgr_0->GenTmpFileName(source_file_name_, "cu");
    if (is_normal_stride_) {
      const char *func_name = "dl_boxes_float4_device_func_global_";
      const char *kernel_name = "dl_boxes_float4_device_func_kernel";

      std::stringstream func_template;
      func_template << "\n#include \"" << plugin_kernel_path.c_str()
                    << "/yolov3_box_kernel.h\"" << std::endl;
      func_template
          << "extern \"C\" __global__ void " << func_name
          << "(const float4* feats_buffer,const float* anchors_buffer,const "
             "int* input_shape_buffer,const int* image_shape_buffer,const int "
             "B, const int H, const int W, const int anchor_num, const int "
             "class_num,float4* boxes_buffer){"
          << std::endl;
      func_template
          << kernel_name
          << "(feats_buffer,anchors_buffer,input_shape_buffer, "
             "image_shape_buffer, B, H, W, anchor_num, class_num,boxes_buffer);"
          << std::endl;
      func_template << "}" << std::endl;

      DebugInfo(func_template.str());
      std::ofstream ofile(source_file_name_, ios::out);
      ofile << func_template.str();
      ofile.close();
    } else {
      const char *func_name = "dl_boxes_float4_device_func_global_aligned_";
      const char *kernel_name = "dl_boxes_float4_device_func_kernel_aligned";

      std::stringstream func_template;
      func_template << "\n#include \"" << plugin_kernel_path.c_str()
                    << "/yolov3_box_kernel.h\"" << std::endl;
      func_template
          << "extern \"C\" __global__ void " << func_name
          << "(const float4* feats_buffer,const float* anchors_buffer,const "
             "int* input_shape_buffer,const int* image_shape_buffer,const int "
             "B, const int H, const int W, const int anchor_num, const int "
             "class_num,float4* boxes_buffer){"
          << std::endl;
      func_template << kernel_name << "(feats_buffer," << s00 << "," << s01
                    << "," << s02 << "," << s03 << ","
                    << "anchors_buffer,input_shape_buffer, image_shape_buffer, "
                       "B, H, W, anchor_num, class_num,boxes_buffer);"
                    << std::endl;
      func_template << "}" << std::endl;

      DebugInfo(func_template.str());
      std::ofstream ofile(source_file_name_, ios::out);
      ofile << func_template.str();
      ofile.close();
    }

    std::unique_ptr<CacheDirMgr> cache_dir_mgr_ =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_->Init();
    std::string file_name;
    bool gen_filename = cache_dir_mgr_->GenTmpFileName(file_name, "fb");
    DebugInfo("Generate file name state: ", gen_filename);
    assert(gen_filename);

    std::string cmd =
        "dlcc -std=c++14 --cuda-gpu-arch=dlgpuc64 --cuda-device-only -w " +
        source_file_name_ + " -o " + file_name;

    DebugInfo(cmd);
    int cmd_flag = system(cmd.c_str());
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
    DebugInfo("call YoloV3BoxPlugin_::GetSerializationSize");
    return (sizeof(is_normal_stride_) + sizeof(align_out_) +
            sizeof(keep_data_len_) + sizeof(int) * keep_data_len_ +
            sizeof(size_t) + sizeof(char) * str_kernel_.size() +
            sizeof(class_num));
  }
  void Terminate() override {}
  void Serialize(void *buffer) const override {
    DebugInfo("call YoloV3BoxPlugin_::Serialize");
    char *bufdata = reinterpret_cast<char *>(buffer);
    size_t size = 0, offset = 0;

    size = sizeof(keep_data_len_);
    memcpy(bufdata + offset, &keep_data_len_, size);
    offset += size;

    size = sizeof(int) * keep_data_len_;
    memcpy(bufdata + offset, keep_data_.data(), size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(is_normal_stride_);
    memcpy(bufdata + offset, &is_normal_stride_, size);
    offset += size;

    size = sizeof(align_out_);
    memcpy(bufdata + offset, &align_out_, size);
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

    size = sizeof(class_num);
    memcpy(bufdata + offset, &class_num, size);
    offset += size;
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
  std::vector<int> keep_data_;
  int keep_data_len_{0};

  int align_out_;
  std::string data_format_;
  std::string str_kernel_;
  bool is_normal_stride_{false};

  int class_num;
};

class YoloV3BoxPluginCreator_ : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call YoloV3BoxPluginCreator_::GetPluginName");
    return "custom_boxes_plugin";  // spatial_transformer_gpu_nne
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call YoloV3BoxPluginCreator_::GetFieldNames");
    mfc_.nbFields = 2;

    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 1) {
        const PluginField fields_ = {"align_out", nullptr, kINT32, 1};  // int32
        v.push_back(fields_);
      } else if (i == 0) {
        const PluginField fields_ = {"data_format", nullptr, kUINT8,
                                     32};  // string ,max length 32 byte
        v.push_back(fields_);
      }
    }

    mfc_.fields = v.data();
    return &mfc_;
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call YoloV3BoxPluginCreator_::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("YoloV3BoxPluginCreator_ create plugin ");

    int align_out;
    std::string data_format;

    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 1) {
        align_out = ((int *)mfc_.fields[i].data)[0];
      }

      if (i == 0) {
        data_format = std::string((char *)(mfc_.fields[i].data));
      }
    }

    if (std::string(name) == GetPluginName()) {
      auto plugin = new YoloV3BoxPlugin_(align_out, data_format);
      DebugInfo("suceess create YoloV3BoxPluginCreator_");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call YoloV3BoxPluginCreator_::DeserializePlugin");
    if (std::string(name) == GetPluginName()) {
      auto plugin = new YoloV3BoxPlugin_(serialData, serialLength);
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call YoloV3BoxPluginCreator_::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call YoloV3BoxPluginCreator_::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

#endif

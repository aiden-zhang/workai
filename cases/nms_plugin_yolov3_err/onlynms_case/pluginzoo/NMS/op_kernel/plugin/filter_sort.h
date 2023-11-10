#pragma once

#include "utils.h"
#include "kernel.h"

using namespace dl::nne;
using namespace std;
using namespace dlNMS;

class FilterSortPlugin : public PluginExt {
 public:
  FilterSortPlugin(float thresh, bool is_ascend, int class_num, int count_num) {
    DebugInfo("construct FilterSortPlugin");

    thresh_ = thresh;
    is_ascend_ = is_ascend;
    class_num_ = class_num;
    count_num_ = count_num;
    inputs_strides_.resize(6);
    output_strides_.resize(3);
    cache_dir_mgr_cu_.Init();
    cache_dir_mgr_fb_.Init();

    // printf("class_num_:%d %d", class_num_, count_num_);

    ser_data_ = 0;
  }

  FilterSortPlugin(const void *data, size_t length) {
    DebugInfo("construct FilterSortPlugin use Serialize data");
    const char *bufdata = reinterpret_cast<char const *>(data);
    size_t size = 0, offset = 0;
    DebugInfo(offset);
    size = sizeof(is_half8_);
    memcpy(&is_half8_, bufdata + offset, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(thresh_);
    memcpy(&thresh_, bufdata + offset, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(is_ascend_);
    memcpy(&is_ascend_, bufdata + offset, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(class_num_);
    memcpy(&class_num_, bufdata + offset, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(count_num_);
    memcpy(&count_num_, bufdata + offset, size);
    offset += size;
    DebugInfo(offset);

    size_t str_size = 0;
    size = sizeof(str_size);
    memcpy(&str_size, bufdata + offset, size);
    offset += size;
    size = sizeof(char) * (str_size);

    str_kernel_.assign(bufdata + offset, size);
    offset += size;

    size = sizeof(num_inputs_);
    memcpy(&num_inputs_, bufdata + offset, size);
    offset += size;
    DebugInfo(offset);

    DebugInfo("deserialize offset", offset);
    ser_data_ = 1;
  }

  ~FilterSortPlugin() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");
    return 3;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    assert((nbInputDims == 6) || (nbInputDims == 1));
    DebugInfo("FilterSortPlugin call GetOutputDimensions");

    if (index == 0) {
      Dims data_dims;
      data_dims.nbDims = 2;
      data_dims.d[0] = class_num_;
      data_dims.d[1] = count_num_;
      return data_dims;
    } else if (index == 1) {
      Dims data_dims;
      data_dims.nbDims = 2;
      data_dims.d[0] = class_num_;
      data_dims.d[1] = count_num_;

      return data_dims;
    } else if (index == 2) {
      Dims data_dims;
      data_dims.nbDims = 1;
      data_dims.d[0] = class_num_;
      return data_dims;
    } else {
      assert(false);
    }
  }

  bool SupportsFormat(const Dims *inputDims, int nbInputs,
                      const Dims *outputDims, int nbOutputs,
                      const DataType *inputTypes, const DataType *outputTypes,
                      Format format) const override {
    return true;
  }

  size_t GetWorkspaceSize(int maxBatchSize) const override {
    DebugInfo("Call GetWorkspaceSize");
    return 0;
  }

  int Enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, void *stream) override {
    DebugInfo("Call Enqueue");
    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);
    CUmodule module{nullptr};
    if (is_half8_) {
      func_name_ = "fused_dl_filter_sort_kernel_720p";
    } else {
      func_name_ = "fused_dl_filter_sort_kernel_416_416";
    }
    std::cout<< "func_name_:"<<func_name_<<std::endl;
    std::cout<< "str_kernel_:"<<str_kernel_<<std::endl;
    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());

    if (err != CUDA_SUCCESS) {
      std::cout << "err: " << err << ", load module failed..." << std::endl;
      return -1;
    }
    assert(err == CUDA_SUCCESS);

    err = cuModuleGetFunction(&func_get_, module, func_name_.c_str());
    DebugInfo("after fused_dl_filter_sort_kernel");
    assert(err == CUDA_SUCCESS);
    if (6 == num_inputs_){
      CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);
      CUdeviceptr input1 = reinterpret_cast<CUdeviceptr>(inputs[1]);
      CUdeviceptr input2 = reinterpret_cast<CUdeviceptr>(inputs[2]);
      CUdeviceptr input3 = reinterpret_cast<CUdeviceptr>(inputs[3]);
      CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(inputs[4]);
      CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(inputs[5]);

      CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
      CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
      CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
      CUstream cu_stream = reinterpret_cast<CUstream>(stream);

      DebugInfo("Thresh: ", thresh_, "Is_ascend", is_ascend_);

      void *args[] = {&input0, &input1,  &input2,  &input3, &input4,
                      &input5, &output0, &output1, &output2};

      int threads_per_block = 128;
      CUDA_KERNEL_NODE_PARAMS params = {func_get_,
                                        (uint32_t)class_num_,
                                        1,
                                        1,
                                        (uint32_t)threads_per_block,
                                        1,
                                        1,
                                        0,
                                        args,
                                        nullptr};
      cuLaunchKernel(func_get_, (uint32_t)class_num_, 1, 1, threads_per_block, 1,
                     1, 0, nullptr, args, nullptr);
      return 0;
    }
    else
    {
      CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);

      CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
      CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
      CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
      CUstream cu_stream = reinterpret_cast<CUstream>(stream);

      DebugInfo("Thresh: ", thresh_, "Is_ascend", is_ascend_);

      void *args[] = {&input0, &output0, &output1, &output2};

      int threads_per_block = 128;
      CUDA_KERNEL_NODE_PARAMS params = {func_get_,
                                        (uint32_t)class_num_,
                                        1,
                                        1,
                                        (uint32_t)threads_per_block,
                                        1,
                                        1,
                                        0,
                                        args,
                                        nullptr};
      cuLaunchKernel(func_get_, (uint32_t)class_num_, 1, 1, threads_per_block, 1,
                     1, 0, nullptr, args, nullptr);
      return 0;
    }
  }

  void *GetGraph(int batchSize, const void *const *inputs, void **outputs,
                 void *workspace, void *stream) override {
    DebugInfo("+FilterSortPlugin::GetGraph");

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);
    CUmodule module{nullptr};
    if (is_half8_) {
      func_name_ = "fused_dl_filter_sort_kernel_720p";
    } else {
      func_name_ = "fused_dl_filter_sort_kernel_416_416";
    }

    CUresult err = cuModuleLoadData(&module, str_kernel_.c_str());

    if (err != CUDA_SUCCESS) {
      std::cout << "err: " << err << ", load module failed..." << std::endl;
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);

    err = cuModuleGetFunction(&func_get_, module, func_name_.c_str());
    DebugInfo("after fused_dl_filter_sort_kernel");
    assert(err == CUDA_SUCCESS);
    if (1 == num_inputs_){
      CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);

      CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
      CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
      CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
      CUstream cu_stream = reinterpret_cast<CUstream>(stream);

      void *args[] = {&input0, &output0, &output1, &output2};

      int threads_per_block = 128;
      CUDA_KERNEL_NODE_PARAMS params = {func_get_,
                                        (uint32_t)class_num_,
                                        1,
                                        1,
                                        (uint32_t)threads_per_block,
                                        1,
                                        1,
                                        0,
                                        args,
                                        nullptr};
      cuGraphCreate(&graph_, 0);
      cuGraphAddKernelNode(&kernel_node_, graph_, nullptr, 0, &params);
      DebugInfo("-FilterSortPlugin::GetGraph");
      return reinterpret_cast<void *>(graph_);
    } else {
      CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);
      CUdeviceptr input1 = reinterpret_cast<CUdeviceptr>(inputs[1]);
      CUdeviceptr input2 = reinterpret_cast<CUdeviceptr>(inputs[2]);
      CUdeviceptr input3 = reinterpret_cast<CUdeviceptr>(inputs[3]);
      CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(inputs[4]);
      CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(inputs[5]);


      CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
      CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
      CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
      CUstream cu_stream = reinterpret_cast<CUstream>(stream);

      DebugInfo("Thresh: ", thresh_, "Is_ascend", is_ascend_, "class_num: ",class_num_);

      void *args[] = {&input0, &input1,  &input2,  &input3, &input4,
                    &input5, &output0, &output1, &output2};


      int threads_per_block = 128;
      CUDA_KERNEL_NODE_PARAMS params = {func_get_,
                                        (uint32_t)class_num_,
                                        1,
                                        1,
                                        (uint32_t)threads_per_block,
                                        1,
                                        1,
                                        0,
                                        args,
                                        nullptr};
      cuGraphCreate(&graph_, 0);
      cuGraphAddKernelNode(&kernel_node_, graph_, nullptr, 0, &params);
      DebugInfo("-FilterSortPlugin::GetGraph");
      return reinterpret_cast<void *>(graph_);
    }
  }

  const char *GetPluginType() const override {
    DebugInfo("call GetPluginType");
    return "custom_filter_sort";
  }

  const char *GetPluginVersion() const override { return "1"; }

  void Destroy() override {
    DebugInfo("call Destroy");
    if (kernel_node_) cuGraphDestroyNode(kernel_node_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    DebugInfo("Clone");
    auto *plugin = new FilterSortPlugin(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("SetPluginNamespace");
    name_space_ = pluginNamespace;
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("Call FilterSortPlugin::GetPluginNamespace");
    return name_space_.data();
  }

  void ConfigurePlugin(PluginTensorDesc const *in, int nbInput,
                       PluginTensorDesc const *out, int nbOutput,
                       int maxBatchSize) override {
    DebugInfo("Call Filter_sort ConfigurePlugin");

    // printf("num_inputs_:%d\n", nbInput);

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

  int Initialize() override {

    DebugInfo("+FilterSortPlugin::Initialize()");
    // std::string plugin_kernel_path = getenv("DLNMS_PLUGIN_KERNEL_PATH");
    DebugInfo("num_inputs_: ",num_inputs_);

    if (6 == num_inputs_)
    {
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
      assert(nbIn1_dims == 4);
      int ss10 = inputs_strides_[1].d[0];
      int ss11 = inputs_strides_[1].d[1];
      int ss12 = inputs_strides_[1].d[2];
      int ss13 = inputs_strides_[1].d[3];
      int s10 = inputs_dims_[1].d[0];
      int s11 = inputs_dims_[1].d[1];
      int s12 = inputs_dims_[1].d[2];
      int s13 = inputs_dims_[1].d[3];

      int nbIn2_dims = inputs_strides_[2].nbDims;
      assert(nbIn2_dims == 4);
      int ss20 = inputs_strides_[2].d[0];
      int ss21 = inputs_strides_[2].d[1];
      int ss22 = inputs_strides_[2].d[2];
      int ss23 = inputs_strides_[2].d[3];
      int s20 = inputs_dims_[2].d[0];
      int s21 = inputs_dims_[2].d[1];
      int s22 = inputs_dims_[2].d[2];
      int s23 = inputs_dims_[2].d[3];

      int nbIn3_dims = inputs_strides_[3].nbDims;
      assert(nbIn3_dims == 4);
      int ss30 = inputs_strides_[3].d[0];
      int ss31 = inputs_strides_[3].d[1];
      int ss32 = inputs_strides_[3].d[2];
      int ss33 = inputs_strides_[3].d[3];
      int s30 = inputs_dims_[3].d[0];
      int s31 = inputs_dims_[3].d[1];
      int s32 = inputs_dims_[3].d[2];
      int s33 = inputs_dims_[3].d[3];

      int nbIn4_dims = inputs_strides_[4].nbDims;
      assert(nbIn4_dims == 4);
      int ss40 = inputs_strides_[4].d[0];
      int ss41 = inputs_strides_[4].d[1];
      int ss42 = inputs_strides_[4].d[2];
      int ss43 = inputs_strides_[4].d[3];
      int s40 = inputs_dims_[4].d[0];
      int s41 = inputs_dims_[4].d[1];
      int s42 = inputs_dims_[4].d[2];
      int s43 = inputs_dims_[4].d[3];

      int nbIn5_dims = inputs_strides_[5].nbDims;
      assert(nbIn5_dims == 4);
      int ss50 = inputs_strides_[5].d[0];
      int ss51 = inputs_strides_[5].d[1];
      int ss52 = inputs_strides_[5].d[2];
      int ss53 = inputs_strides_[5].d[3];
      int s50 = inputs_dims_[5].d[0];
      int s51 = inputs_dims_[5].d[1];
      int s52 = inputs_dims_[5].d[2];
      int s53 = inputs_dims_[5].d[3];

      int nbOut0_dims = output_strides_[0].nbDims;
      assert(nbOut0_dims == 2);
      int out_ss00 = output_strides_[0].d[0];
      int out_ss01 = output_strides_[0].d[1];
      int out_s00 = outputs_dims_[0].d[0];
      int out_s01 = outputs_dims_[0].d[1];

      int nbOut1_dims = output_strides_[1].nbDims;
      assert(nbOut1_dims == 2);
      int out_ss10 = output_strides_[1].d[0];
      int out_ss11 = output_strides_[1].d[1];
      int out_s10 = outputs_dims_[1].d[0];
      int out_s11 = outputs_dims_[1].d[1];

      int nbOut2_dims = output_strides_[2].nbDims;
      assert(nbOut2_dims == 1);
      int out_ss20 = output_strides_[2].d[0];
      int out_s20 = outputs_dims_[2].d[0];

      DebugInfo("in0: ", s00, s01, s02, s03, ss00, ss01, ss02, ss03);
      DebugInfo("in1: ", s10, s11, s12, s13, ss10, ss11, ss12, ss13);
      DebugInfo("in2: ", s20, s21, s22, s23, ss20, ss21, ss22, ss23);
      DebugInfo("in3: ", s30, s31, s32, s33, ss30, ss31, ss32, ss33);
      DebugInfo("in4: ", s40, s41, s42, s43, ss40, ss41, ss42, ss43);
      DebugInfo("in5: ", s50, s51, s52, s53, ss50, ss51, ss52, ss53);

      DebugInfo("out0: ", out_s00, out_s01, out_ss00, out_ss01);
      DebugInfo("out1: ", out_s10, out_s11, out_ss10, out_ss11);
      DebugInfo("out2: ", out_s20, out_ss20);

      if (ss02 % 8 == 0 && ss12 % 8 == 0 && ss22 % 8 == 0 && ss32 % 8 == 0 &&
          ss42 % 8 == 0 && ss52 % 8 == 0 && s03 % 8 == 0 && s13 % 8 == 0 &&
          s23 % 8 == 0 && s33 % 8 == 0 && s43 % 8 == 0 && s53 % 8 == 0)
        is_half8_ = true;

      std::string func_name;
      std::string kernel_name;
      std::string source_file;

      if (is_half8_) {
        func_name_ = "fused_dl_filter_sort_kernel_720p";
        kernel_name = "fused_dl_filter_sort_half8_kernel";
      } else {
        func_name_ = "fused_dl_filter_sort_kernel_416_416";
        kernel_name = "fused_dl_filter_sort_kernel";
      }

      std::stringstream func_template;
      // func_template << "\n#include \"" << plugin_kernel_path.c_str()
      //               << "/filter_sort_kernel.h\"" << std::endl;
      func_template << filter_sort_src_file << std::endl;
      func_template
          << "extern \"C\" __global__ void " << func_name_.c_str()
          << "(half* __restrict__ in0, half* __restrict__ in1, half* "
             "__restrict__ in2, half* __restrict__ in3, half* __restrict__ in4, "
             "half* __restrict__ in5, float* __restrict__ out0, int* "
             "__restrict__ out1, int* __restrict__ out2){"
          << std::endl;

      func_template << kernel_name.c_str() << "(in0, " << s00 << ", " << s01
                    << ", " << s02 << ", " << s03 << ", " << ss00 << ", " << ss01
                    << ", " << ss02 << ", " << ss03 << ", "
                    << "in1, " << s10 << ", " << s11 << ", " << s12 << ", " << s13
                    << ", " << ss10 << ", " << ss11 << ", " << ss12 << ", "
                    << ss13 << ", "
                    << "in2, " << s20 << ", " << s21 << ", " << s22 << ", " << s23
                    << ", " << ss20 << ", " << ss21 << ", " << ss22 << ", "
                    << ss23 << ", "
                    << "in3, " << s30 << ", " << s31 << ", " << s32 << ", " << s33
                    << ", " << ss30 << ", " << ss31 << ", " << ss32 << ", "
                    << ss33 << ", "
                    << "in4, " << s40 << ", " << s41 << ", " << s42 << ", " << s43
                    << ", " << ss40 << ", " << ss41 << ", " << ss42 << ", "
                    << ss43 << ", "
                    << "in5, " << s50 << ", " << s51 << ", " << s52 << ", " << s53
                    << ", " << ss50 << ", " << ss51 << ", " << ss52 << ", "
                    << ss53 << ", "
                    << "out0, " << out_s00 << ", " << out_s01 << ", " << out_ss00
                    << ", " << out_ss01 << ", "
                    << "out1, " << out_s10 << ", " << out_s11 << ", " << out_ss10
                    << ", " << out_ss11 << ", "
                    << "out2, " << out_s20 << ", " << out_ss20 << ", " << thresh_
                    << ", " << is_ascend_ << ");" << std::endl;

      func_template << "}" << std::endl;
      bool gen_cu = cache_dir_mgr_cu_.GenTmpFileName(source_file, "cu");

      DebugInfo(func_template.str());
      std::ofstream ofile(source_file, ios::out);
      ofile << func_template.str();
      ofile.close();

      bool gen_filename = cache_dir_mgr_fb_.GenTmpFileName(kernel_file_, "fb");
      DebugInfo("Generate file name state: ", gen_filename);
      assert(gen_filename);

      std::string cmd =
          "dlcc -std=c++14 --cuda-gpu-arch=dlgpuc64 --cuda-device-only " +
          source_file + " -o " + kernel_file_;

      DebugInfo(cmd);
      int cmd_flag = PluginCallSystem(cmd.c_str());
      assert(cmd_flag != 0);

      // serialize bc file to string
      if (SaveBcToString(kernel_file_, str_kernel_)) {
        DebugInfo("SaveBcToString");
      } else {
        assert(false);
      }
    } else {
      int nbIn0_dims = inputs_strides_[0].nbDims;
      assert(nbIn0_dims == 2);
      int ss00 = inputs_strides_[0].d[0];
      int ss01 = inputs_strides_[0].d[1];
      
      int s00 = inputs_dims_[0].d[0];
      int s01 = inputs_dims_[0].d[1];
      
      int nbOut0_dims = output_strides_[0].nbDims;
      assert(nbOut0_dims == 2);
      int out_ss00 = output_strides_[0].d[0];
      int out_ss01 = output_strides_[0].d[1];
      int out_s00 = outputs_dims_[0].d[0];
      int out_s01 = outputs_dims_[0].d[1];

      int nbOut1_dims = output_strides_[1].nbDims;
      assert(nbOut1_dims == 2);
      int out_ss10 = output_strides_[1].d[0];
      int out_ss11 = output_strides_[1].d[1];
      int out_s10 = outputs_dims_[1].d[0];
      int out_s11 = outputs_dims_[1].d[1];

      int nbOut2_dims = output_strides_[2].nbDims;
      assert(nbOut2_dims == 1);
      int out_ss20 = output_strides_[2].d[0];
      int out_s20 = outputs_dims_[2].d[0];

      DebugInfo("in0: ", s00, s01, ss00, ss01);

      DebugInfo("out0: ", out_s00, out_s01, out_ss00, out_ss01);
      DebugInfo("out1: ", out_s10, out_s11, out_ss10, out_ss11);
      DebugInfo("out2: ", out_s20, out_ss20);

      is_half8_ = false;

      std::string func_name;
      std::string kernel_name;
      std::string source_file;

      if (is_half8_) {
        func_name_ = "fused_dl_filter_sort_kernel_720p";
        kernel_name = "fused_dl_filter_sort_half8_pure_kernel";
      } else {
        func_name_ = "fused_dl_filter_sort_kernel_416_416";
        kernel_name = "fused_dl_filter_sort_pure_kernel";
      }

      std::stringstream func_template;
      // func_template << "\n#include \"" << plugin_kernel_path.c_str()
      //               << "/filter_sort_kernel.h\"" << std::endl;
      func_template << filter_sort_src_file << std::endl;
      func_template
          << "extern \"C\" __global__ void " << func_name_.c_str()
          << "(float* __restrict__ in0,"
             "float* __restrict__ out0,"
             "int* __restrict__ out1, int* __restrict__ out2){"
          << std::endl;

      func_template << kernel_name.c_str() << "(in0, " << s00 << ", " << s01 << ", "
                    << ss00 << ", " << ss01 << ", "
                    << "out0, " << out_s00 << ", " << out_s01 << ", " << out_ss00
                    << ", " << out_ss01 << ", "
                    << "out1, " << out_s10 << ", " << out_s11 << ", " << out_ss10
                    << ", " << out_ss11 << ", "
                    << "out2, " << out_s20 << ", " << out_ss20 << ", " << thresh_
                    << ", " << is_ascend_ << ");" << std::endl;

      func_template << "}" << std::endl;
      bool gen_cu = cache_dir_mgr_cu_.GenTmpFileName(source_file, "cu");

      DebugInfo(func_template.str());
      std::ofstream ofile(source_file, ios::out);
      ofile << func_template.str();
      ofile.close();

      bool gen_filename = cache_dir_mgr_fb_.GenTmpFileName(kernel_file_, "fb");
      DebugInfo("Generate file name state: ", gen_filename);
      assert(gen_filename);

      std::string cmd =
          "dlcc -std=c++14 --cuda-gpu-arch=dlgpuc64 --cuda-device-only " +
          source_file + " -o " + kernel_file_;

      DebugInfo(cmd);
      int cmd_flag = PluginCallSystem(cmd.c_str());
      assert(cmd_flag != 0);

      // serialize bc file to string
      if (SaveBcToString(kernel_file_, str_kernel_)) {
        DebugInfo("SaveBcToString");
      } else {
        assert(false);
      }
    }
    DebugInfo("-FilterSortPlugin::Initialize()");
    return 0;
  }
  size_t GetSerializationSize() const override {
    DebugInfo("call FilterSortPlugin::GetSerializationSize");
    return sizeof(is_half8_) + sizeof(thresh_) + sizeof(is_ascend_) +
           sizeof(class_num_) + sizeof(count_num_) + sizeof(size_t) +
           sizeof(char) * str_kernel_.size() + sizeof(num_inputs_);
  }
  void Terminate() override {}
  void Serialize(void *buffer) const override {
    DebugInfo("call FilterSortPlugin::Serialize");
    char *bufdata = reinterpret_cast<char *>(buffer);
    size_t size = 0, offset = 0;
    DebugInfo(offset);

    size = sizeof(is_half8_);
    memcpy(bufdata + offset, &is_half8_, size);
    offset += size;

    size = sizeof(thresh_);
    memcpy(bufdata + offset, &thresh_, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(is_ascend_);
    memcpy(bufdata + offset, &is_ascend_, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(class_num_);
    memcpy(bufdata + offset, &class_num_, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(count_num_);
    memcpy(bufdata + offset, &count_num_, size);
    offset += size;
    DebugInfo(offset);

    size_t str_len = str_kernel_.size();
    size = sizeof(str_len);
    memcpy(bufdata + offset, &str_len, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(char) * (str_len);
    memcpy(bufdata + offset, str_kernel_.c_str(), size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(num_inputs_);
    memcpy(bufdata + offset, &num_inputs_, size);
    offset += size;
    DebugInfo(offset);

    DebugInfo("call FilterSortPlugin::Serialize", offset);
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
  std::string kernel_file_;
  std::string func_name_;

  CacheDirMgr cache_dir_mgr_cu_;
  CacheDirMgr cache_dir_mgr_fb_;

  float thresh_;
  bool is_ascend_;
  int class_num_;
  int count_num_;
  std::string str_kernel_;
  bool is_half8_{false};
};

class FilterSortPluginCreator : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call FilterSortPluginCreator::GetPluginName");
    return "custom_filter_sort";
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call FilterSortPluginCreator::GetFieldNames");
    mfc_.nbFields = 4;

    for (int i = 0; i < mfc_.nbFields; ++i) {
      if (i == 0) {
        const PluginField fields_ = {"threshold", nullptr, kFLOAT32, 1};
        v.push_back(fields_);
      } else if (i == 1) {
        const PluginField fields_ = {"is_ascend", nullptr, kUINT8, 1};
        v.push_back(fields_);
      } else if (i == 2) {
        const PluginField fields_ = {"class_num", nullptr, kINT32, 1};
        v.push_back(fields_);
      } else if (i == 3) {
        const PluginField fields_ = {"count_num", nullptr, kINT32, 1};
        v.push_back(fields_);
      }
    }
    mfc_.fields = v.data();
    return &mfc_;
  }

  const char *GetPluginVersion() const override {
    DebugInfo("call FilterSortPluginCreator::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("FilterSortPluginCreator create plugin ");

    float Thresh = ((float *)mfc_.fields[0].data)[0];
    bool Is_ascend = ((bool *)mfc_.fields[1].data)[0];
    int Class_num = ((int *)mfc_.fields[2].data)[0];
    int Count_num = ((int *)mfc_.fields[3].data)[0];
    //printf("-------++++++++:%s %d %d %d %d\n", name, Thresh, Is_ascend, Class_num, Count_num);
    if ((std::string(name) == GetPluginName()) ||
        (std::string(name) == "dl.filter_sort") 
         //||(std::string(name) == "custom_filter_sort")
        ) {
          
      auto plugin = new FilterSortPlugin(Thresh, Is_ascend, Class_num, Count_num);
      DebugInfo("suceess create FilterSortPluginCreator");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call FilterSortPluginCreator::DeserializePlugin");
    if (std::string(name) == GetPluginName() ||
        std::string(name) == "dl.filter_sort" ||
        std::string(name) == "custom_filter_sort") {
      auto plugin = new FilterSortPlugin(serialData, serialLength);
      DebugInfo("suceess create FilterSortPluginCreator use serial data");
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call FilterSortPluginCreator::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call FilterSortPluginCreator::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

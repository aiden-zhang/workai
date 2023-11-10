#pragma once

#include "utils.h"

using namespace dl::nne;
using namespace std;

class FilterSortSingleInputPlugin : public Plugin {
 public:
  FilterSortSingleInputPlugin(float thresh, bool is_ascend, int class_num, int count_num) {
    DebugInfo("construct FilterSortSingleInputPlugin");

    Thresh = thresh;
    Is_ascend = is_ascend;
    Class_num = class_num;
    Count_num = count_num;
    m_inputDims.resize(2);
    m_outputDims.resize(3);
    serData = 1;
  }

  FilterSortSingleInputPlugin(const void *data, size_t length) {
    DebugInfo("construct FilterSortSingleInputPlugin use Serialize data");
    const char *bufdata = reinterpret_cast<char const *>(data);
    size_t size = 0, offset = 0;

    size = sizeof(Thresh);
    memcpy(&Thresh, bufdata + offset, size);
    offset += size;

    size = sizeof(Is_ascend);
    memcpy(&Is_ascend, bufdata + offset, size);
    offset += size;

    size = sizeof(Class_num);
    memcpy(&Class_num, bufdata + offset, size);
    offset += size;

    size = sizeof(Count_num);
    memcpy(&Count_num, bufdata + offset, size);
    offset += size;

    size = sizeof(s00);
    memcpy(&s00, bufdata + offset, size);
    offset += size;
    size = sizeof(s01);
    memcpy(&s01, bufdata + offset, size);
    offset += size;
    size = sizeof(s02);
    memcpy(&s02, bufdata + offset, size);
    offset += size;
    size = sizeof(s03);
    memcpy(&s03, bufdata + offset, size);
    offset += size;
    size = sizeof(s10);
    memcpy(&s10, bufdata + offset, size);
    offset += size;
    size = sizeof(s11);
    memcpy(&s11, bufdata + offset, size);
    offset += size;
    size = sizeof(s12);
    memcpy(&s12, bufdata + offset, size);
    offset += size;
    size = sizeof(s13);
    memcpy(&s13, bufdata + offset, size);
    offset += size;

    size = sizeof(out_s00);
    memcpy(&out_s00, bufdata + offset, size);
    offset += size;
    size = sizeof(out_s01);
    memcpy(&out_s01, bufdata + offset, size);
    offset += size;
    size = sizeof(out_s10);
    memcpy(&out_s10, bufdata + offset, size);
    offset += size;
    size = sizeof(out_s11);
    memcpy(&out_s11, bufdata + offset, size);
    offset += size;
    size = sizeof(out_s20);
    memcpy(&out_s20, bufdata + offset, size);
    offset += size;

    DebugInfo(offset);

    m_inputDims.resize(2);
    m_outputDims.resize(3);
    serData = 1;
  }

  ~FilterSortSingleInputPlugin() {}

  int GetNbOutputs() const override {
    DebugInfo("call GetNbOutputs");
    return 3;
  }

  Dims GetOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    assert(nbInputDims == 2);
    DebugInfo("call GetOutputDimensions");

    if (index == 0) {
      Dims data_dims;
      data_dims.nbDims = 2;
      data_dims.d[0] = Class_num;
      data_dims.d[1] = Count_num;
      return data_dims;
    } else if (index == 1) {
      Dims data_dims;
      data_dims.nbDims = 2;
      data_dims.d[0] = Class_num;
      data_dims.d[1] = Count_num;

      return data_dims;
    } else if (index == 2) {
      Dims data_dims;
      data_dims.nbDims = 1;
      data_dims.d[0] = Class_num;
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

  int Enqueue(int batchSize, const void *const *inputs,
              const void *const *input_stride, void **outputs,
              const void *const *output_stride, void *workspace,
              void *stream) override {
    DebugInfo("call Enqueue");
    return 0;
  }

  void *GetGraph(int batchSize, const void *const *inputs,
                 const void *const *input_stride, void **outputs,
                 const void *const *output_stride, void *workspace,
                 void *stream) override {
    DebugInfo("filter_sort_single_input call GetGraph");

    std::string plugin_kernel_path = getenv("YOLOV2_PLUGIN_KERNEL_PATH");

    int ss00 = ((int *)input_stride[0])[0];
    int ss01 = ((int *)input_stride[0])[1];
    int ss02 = ((int *)input_stride[0])[2];
    int ss03 = ((int *)input_stride[0])[3];

    int ss10 = ((int *)input_stride[1])[0];
    int ss11 = ((int *)input_stride[1])[1];
    int ss12 = ((int *)input_stride[1])[2];
    int ss13 = ((int *)input_stride[1])[3];

    int out_ss00 = ((int *)output_stride[0])[0];
    int out_ss01 = ((int *)output_stride[0])[1];
    int out_ss10 = ((int *)output_stride[1])[0];
    int out_ss11 = ((int *)output_stride[1])[1];
    int out_ss20 = ((int *)output_stride[2])[0];

    bool is_half8 = false;
    if (ss02 % 8 == 0 && ss12 % 8 == 0 &&
        s03 % 8 == 0 && s13 % 8 == 0)
      is_half8 = true;

    std::string file_name;
    const char *func_name = "filter_sort_single_input_kernel_416_416";
    const char *kernel_name = "filter_sort_single_input_kernel";

    std::unique_ptr<CacheDirMgr> cache_dir_mgr_0 =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_0->Init();
    std::string source_file;
    std::unique_ptr<CacheDirMgr> cache_dir_mgr_ =
        std::make_unique<CacheDirMgr>();
    cache_dir_mgr_->Init();

    if (is_half8) {
      func_name = "filter_sort_single_input_kernel_720p";
      kernel_name = "filter_sort_single_input_half8_kernel";
      DebugInfo("file_name: ", file_name);
    }

    std::stringstream func_template;
    func_template << "\n#include \"" << plugin_kernel_path.c_str()
                  << "/filter_sort_kernel.h\"" << std::endl;
    func_template
        << "extern \"C\" __global__ void " << func_name
        << "(half* __restrict__ in0, half* __restrict__ in1,"
           "float* __restrict__ out0, int* "
           "__restrict__ out1, int* __restrict__ out2){"
        << std::endl;

    func_template << kernel_name << "(in0, " << s00 << ", " << s01 << ", "
                  << s02 << ", " << s03 << ", " << ss00 << ", " << ss01 << ", "
                  << ss02 << ", " << ss03 << ", "
                  << "in1, " << s10 << ", " << s11 << ", " << s12 << ", " << s13
                  << ", " << ss10 << ", " << ss11 << ", " << ss12 << ", "
                  << ss13 << ", "
                  << "out0, " << out_s00 << ", " << out_s01 << ", " << out_ss00
                  << ", " << out_ss01 << ", "
                  << "out1, " << out_s10 << ", " << out_s11 << ", " << out_ss10
                  << ", " << out_ss11 << ", "
                  << "out2, " << out_s20 << ", " << out_ss20 << ", " << Thresh
                  << ", " << Is_ascend << ");" << std::endl;

    func_template << "}" << std::endl;
    bool gen_cu = cache_dir_mgr_0->GenTmpFileName(source_file, "cu");

    DebugInfo(func_template.str());
    std::ofstream ofile(source_file, ios::out);
    ofile << func_template.str();
    ofile.close();

    bool gen_filename = cache_dir_mgr_->GenTmpFileName(file_name, "bc");
    DebugInfo("Generate file name state: ", gen_filename);
    assert(gen_filename);

    std::string cmd =
        "dlcc -std=c++14 --cuda-gpu-arch=dlgpuc64 --cuda-device-only " +
        source_file + " -o " + file_name;

    DebugInfo(cmd);
    if (system(cmd.c_str())) {
      return nullptr;
    }

    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);

    CUmodule module{nullptr};
    CUresult err = cuModuleLoad(&module, file_name.c_str());

    if (err != CUDA_SUCCESS) {
      std::cout << "err: " << err << ", load module failed..." << std::endl;
      return nullptr;
    }
    assert(err == CUDA_SUCCESS);

    err = cuModuleGetFunction(&func_get, module, func_name);
    DebugInfo("after filter_sort_single_input_kernel");
    assert(err == CUDA_SUCCESS);
    CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(inputs[0]);
    CUdeviceptr input1 = reinterpret_cast<CUdeviceptr>(inputs[1]);

    CUdeviceptr output0 = reinterpret_cast<CUdeviceptr>(outputs[0]);
    CUdeviceptr output1 = reinterpret_cast<CUdeviceptr>(outputs[1]);
    CUdeviceptr output2 = reinterpret_cast<CUdeviceptr>(outputs[2]);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    DebugInfo("in0: ", s00, s01, s02, s03, ss00, ss01, ss02, ss03);
    DebugInfo("in1: ", s10, s11, s12, s13, ss10, ss11, ss12, ss13);

    DebugInfo("out0: ", out_s00, out_s01, out_ss00, out_ss01);
    DebugInfo("out1: ", out_s10, out_s11, out_ss10, out_ss11);
    DebugInfo("out2: ", out_s20, out_ss20);

    DebugInfo("Thresh: ", Thresh, "Is_ascend", Is_ascend);

    void *args[] = {&input0, &input1, &output0, &output1, &output2};

    int threads_per_block = 128;
    CUDA_KERNEL_NODE_PARAMS params = {func_get,
                                      (uint32_t)Class_num,
                                      1,
                                      1,
                                      (uint32_t)threads_per_block,
                                      1,
                                      1,
                                      0,
                                      args,
                                      nullptr};
    cuGraphCreate(&graph_, 0);
    cuGraphAddKernelNode(&kernelNode_, graph_, nullptr, 0, &params);

    return reinterpret_cast<void *>(graph_);
  }

  void ConfigureWithFormat(const Dims *inputDims, int nbInputs,
                           const Dims *outputDims, int nbOutputs,
                           const DataType *inputTypes,
                           const DataType *outputTypes, Format format,
                           int maxBatchSize) override {
    DebugInfo("call ConfigureWithFormat");
    assert(nbInputs == 2);
    assert(nbOutputs == 3);
    for (int i = 0; i < nbInputs; ++i) {
      m_inputDims[i].nbDims = inputDims[i].nbDims;
      for (int j = 0; j < inputDims[i].nbDims; ++j) {
        m_inputDims[i].d[j] = inputDims[i].d[j];
      }
    }
    s00 = m_inputDims[0].d[0];
    s01 = m_inputDims[0].d[1];
    s02 = m_inputDims[0].d[2];
    s03 = m_inputDims[0].d[3];

    s10 = m_inputDims[1].d[0];
    s11 = m_inputDims[1].d[1];
    s12 = m_inputDims[1].d[2];
    s13 = m_inputDims[1].d[3];


    for (int i = 0; i < nbOutputs; i++) {
      for (int j = 0; j < outputDims[i].nbDims; j++) {
        m_outputDims[i].d[j] = outputDims[i].d[j];
      }
    }

    out_s00 = m_outputDims[0].d[0];
    out_s01 = m_outputDims[0].d[1];
    out_s10 = m_outputDims[1].d[0];
    out_s11 = m_outputDims[1].d[1];
    out_s20 = m_outputDims[2].d[0];
  }

  const char *GetPluginType() const override {
    DebugInfo("call GetPluginType");
    return "custom_filter_sort_single_input";
  }

  const char *GetPluginVersion() const override { return "1"; }

  void Destroy() override {
    DebugInfo("call Destroy");
    if (kernelNode_) cuGraphDestroyNode(kernelNode_);
    if (graph_) cuGraphDestroy(graph_);
    delete this;
  }

  Plugin *Clone() const override {
    DebugInfo("Clone");
    auto *plugin = new FilterSortSingleInputPlugin(*this);
    return plugin;
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("SetPluginNamespace");
    mNamespace = pluginNamespace;
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call FilterSortSingleInputPlugin::GetPluginNamespace");
    return mNamespace.data();
  }

  int Initialize() override { return 0; }
  size_t GetSerializationSize() const override {
    DebugInfo("call FilterSortSingleInputPlugin::GetSerializationSize");
    return (sizeof(Thresh) + sizeof(Is_ascend) + sizeof(Class_num) +
            sizeof(Count_num) + sizeof(s00) + sizeof(s01) + sizeof(s02) +
            sizeof(s03) + sizeof(s10) + sizeof(s11) + sizeof(s12) +
            sizeof(s13) + sizeof(out_s00) + sizeof(out_s01) + sizeof(out_s10) +
            sizeof(out_s11) + sizeof(out_s20));
  }
  void Terminate() override {}
  void Serialize(void *buffer) const override {
    DebugInfo("call FilterSortSingleInputPlugin::Serialize");
    char *bufdata = reinterpret_cast<char *>(buffer);
    size_t size = 0, offset = 0;
    DebugInfo(offset);

    size = sizeof(Thresh);
    memcpy(bufdata + offset, &Thresh, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(Is_ascend);
    memcpy(bufdata + offset, &Is_ascend, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(Class_num);
    memcpy(bufdata + offset, &Class_num, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(Count_num);
    memcpy(bufdata + offset, &Count_num, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s00);
    memcpy(bufdata + offset, &s00, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s01);
    memcpy(bufdata + offset, &s01, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s02);
    memcpy(bufdata + offset, &s02, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s03);
    memcpy(bufdata + offset, &s03, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s10);
    memcpy(bufdata + offset, &s10, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s11);
    memcpy(bufdata + offset, &s11, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s12);
    memcpy(bufdata + offset, &s12, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(s13);
    memcpy(bufdata + offset, &s13, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(out_s00);
    memcpy(bufdata + offset, &out_s00, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(out_s01);
    memcpy(bufdata + offset, &out_s01, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(out_s10);
    memcpy(bufdata + offset, &out_s10, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(out_s11);
    memcpy(bufdata + offset, &out_s11, size);
    offset += size;
    DebugInfo(offset);

    size = sizeof(out_s20);
    memcpy(bufdata + offset, &out_s20, size);
    offset += size;
    DebugInfo(offset);
  }

 private:
  std::string mNamespace;
  int outputSize{0};
  CUfunction func_get{};
  int serData{0};
  CUgraph graph_{nullptr};
  CUgraphNode kernelNode_{nullptr};
  std::vector<Dims> m_inputDims;
  std::vector<Dims> m_outputDims;
  CUmodule module{nullptr};

  float Thresh;
  bool Is_ascend;
  int Class_num;
  int Count_num;
  int s00{0};
  int s01{0};
  int s02{0};
  int s03{0};

  int s10{0};
  int s11{0};
  int s12{0};
  int s13{0};

  int out_s00{0};
  int out_s01{0};
  int out_s10{0};
  int out_s11{0};
  int out_s20{0};
};

class FilterSortSingleInputPluginCreator : public PluginCreator {
 public:
  const char *GetPluginName() const override {
    DebugInfo("call FilterSortSingleInputPluginCreator::GetPluginName");
    return "custom_filter_sort_single_input";
  }

  const PluginFieldCollection *GetFieldNames() override {
    DebugInfo("call FilterSortSingleInputPluginCreator::GetFieldNames");
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
    DebugInfo("call FilterSortSingleInputPluginCreator::GetPluginVersion");
    return "1";
  }

  Plugin *CreatePlugin(const char *name,
                       const PluginFieldCollection *fc) override {
    DebugInfo("FilterSortSingleInputPluginCreator create plugin ");

    float Thresh = ((float *)mfc_.fields[0].data)[0];
    bool Is_ascend = ((bool *)mfc_.fields[1].data)[0];
    int Class_num = ((int *)mfc_.fields[2].data)[0];
    int Count_num = ((int *)mfc_.fields[3].data)[0];
    if ((std::string(name) == GetPluginName()) ||
        (std::string(name) == "custom_filter_sort_single_input")) {
      auto plugin =
          new FilterSortSingleInputPlugin(Thresh, Is_ascend, Class_num, Count_num);
      DebugInfo("suceess create FilterSortSingleInputPluginCreator");
      return plugin;
    } else {
      return nullptr;
    }
  }

  Plugin *DeserializePlugin(const char *name, const void *serialData,
                            size_t serialLength) override {
    DebugInfo("call FilterSortSingleInputPluginCreator::DeserializePlugin");
    if (std::string(name) == GetPluginName() ||
        std::string(name) == "custom_filter_sort_single_input") {
      auto plugin = new FilterSortSingleInputPlugin(serialData, serialLength);
      DebugInfo("suceess create FilterSortSingleInputPluginCreator use serial data");
      return plugin;
    } else {
      return nullptr;
    }
  }

  void SetPluginNamespace(const char *pluginNamespace) override {
    DebugInfo("call FilterSortSingleInputPluginCreator::SetPluginNamespace");
  }

  const char *GetPluginNamespace() const override {
    DebugInfo("call FilterSortSingleInputPluginCreator::GetPluginNamespace");
    return mNamespace;
  }

  static constexpr char *mNamespace{"dlnne"};

 private:
  std::vector<PluginField> v;
  PluginFieldCollection mfc_;
};

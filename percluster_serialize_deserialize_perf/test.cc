
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlnne.h>
#include "stdlib.h"
// #include "dlcuda/dl_packet_frame.h"

using namespace dl::nne;
#define CHECK(call)                                                \
    do {                                                           \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess) {                                \
            fprintf(stderr, "Error:%s:%d,", __FILE__, __LINE__);   \
            fprintf(stderr, "error code :%s, reason: %d\n",error , cudaGetErrorString(error));\
            exit(1);                                               \
        }                                                          \
    } while(0)

#define TEST_RETURN_FALSE_EXPR(expr)                               \
    do {                                                           \
        const bool ret_value = (expr);                               \
        if ( !ret_value ) {                                        \
            printf("Call:%s  fail in %s at %d line.",#expr, __FILE__, __LINE__);\
            return ret_value;                                      \
        }                                                          \
    } while(0)

class Bindings
{
public:
     Bindings(Dims shape, DataType data_type, int batch_size, int index, std::string name,bool is_input)
    : shape(shape), dtype(data_type), index(index), batch_size(batch_size), node_name(name), is_input(is_input) {};

    int get_batched_node_size()
    {
        int len = 1;
        for (int i = 0; i < shape.nbDims; i++) {
            len *= shape.d[i];
        }
        len *= (batch_size * sizeof(float));
        return len;
    }
    ~Bindings() = default;

public:
    float* data = nullptr;
    bool is_input;
private: 
    Dims shape;
    DataType   dtype;
    int batch_size;
    int index;
    std::string node_name;
    

};


int main(int argc, char **argv)
{
    if(argc < 2) {
        std::cout<<"param err!!"<<std::endl;
        return -1;
    }
    int is_weight_shared = atoi(argv[1]);
    std::string inputdata_path = "../networkinputdata_fp32.data";
    std::string model_path("../yolov5s_bad.downcast.rlym");//fp16
    // std::string model_path("../yolov5s_bad.quantized.rlym");//int8
    // std::string model_path("/mercury/share/DLI/models/customer/weicheng/yolov5s_bad.onnx");//fp32
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int cluster_count = prop.clusterCount;

    auto builder = CreateInferBuilder();
    auto network = builder->CreateNetwork();
    int batch_size = 16;
    BuilderConfig builder_cfg;
    builder_cfg.callback = nullptr;
    builder_cfg.max_batch_size = batch_size;
    builder_cfg.dump_dot = false;
    builder_cfg.print_profiling = false;
    auto weight_share_mode = static_cast<WeightShareMode>(cluster_count);
    builder_cfg.ws_mode = weight_share_mode;
    if( is_weight_shared ) {
        builder->SetBuilderConfig(builder_cfg);
    }

    builder->SetMaxBatchSize(batch_size);
    auto parser = dl::nne::CreateParser();
    parser->Parse(model_path.c_str(),*network);
    Engine* engine = builder->BuildEngine(*network);
    ClusterConfig config = weight_share_mode == 4 ? kCluster0123:weight_share_mode == 2 ? kCluster01:kCluster0;
    ExecutionContext* context = nullptr;
    if( is_weight_shared ) {
        context = engine->CreateExecutionContext(config);
    } else {
        context = engine->CreateExecutionContext();
    }

    /*read input data to host*/
    std::ifstream finput(inputdata_path);
    finput.seekg(0, std::ios::end);
    uint64_t input_length = static_cast<uint64_t>(finput.tellg());
    finput.seekg(0, std::ios::beg);


    float* input_host = (float*)malloc(input_length);
    finput.read((char*)input_host, static_cast<int64_t>(input_length));
    finput.close();


    /* create bindings */
    auto nb_bindings = engine->GetNbBindings();
    std::vector<Bindings*> binding_list;
    void **binding_device_array = (void**)malloc(nb_bindings * sizeof(void*));
    void **binding_host_array = (void**)malloc(nb_bindings * sizeof(void*));
    for(auto i = 0; i < nb_bindings; i++) {
        auto name = engine->GetBindingName(i);
        auto shape = engine->GetBindingDimensions(i);
        auto data_type = engine->GetBindingDataType(i);
        bool is_input = engine->BindingIsInput(i);
        auto binding = new Bindings(shape, data_type, batch_size, i, name, is_input);

        float* pdev = nullptr;
        float* phost = nullptr;
        int buf_len = binding->get_batched_node_size();
        CHECK( cudaMalloc(&pdev, buf_len));
        binding->data = (float*)pdev;
        binding_device_array[i] = pdev;

        if ( !is_input ) {
            phost = (float*)malloc(buf_len);
            binding_host_array[i]   = phost; //host涓婂瓨鏀炬渶缁堢殑杈撳嚭; 杈撳叆鏁版嵁涔熷彲浠ョ洿鎺ユ斁鍒拌繖閲?       
             }
        
        binding_list.push_back(binding); 
    }

    /* copy inputdata from host  to device */
    for(auto i = 0; i < nb_bindings; i++) {
        if( binding_list[i]->is_input ) {
            for( auto j = 0; j < batch_size; j++) {
                float* dest = binding_list[i]->data + j * sizeof(float) * (binding_list[i]->get_batched_node_size()/batch_size);
                cudaMemcpy((void*)dest, (void*)input_host, input_length, cudaMemcpyHostToDevice); 
            }
            
        }
    }

    /*do infer*/
    context->Execute(batch_size, binding_device_array);

    /* copy output from device to host */
    std::string output_file;
    if(is_weight_shared) {
        output_file = "../out_weightshared.data";
    } else {
        output_file = "../out_singlecluster.data";
    }
    std::ofstream fout(output_file, std::ios::binary);

    for(auto i = 0; i < nb_bindings; i++) {
        if( !binding_list[i]->is_input ) { //only test one output node
            int outputdata_len = binding_list[i]->get_batched_node_size();
            std::cout << "output_len:"<< outputdata_len <<std::endl;
            cudaMemcpy(binding_host_array[i], binding_list[i]->data, outputdata_len, cudaMemcpyDeviceToHost);
            fout.write((char *)binding_host_array[i], outputdata_len/batch_size); //just take one of batchsized output
        }
    }

   fout.close();
    return 0;
}
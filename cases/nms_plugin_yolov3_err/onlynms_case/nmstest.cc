#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cu_ext.h>
#include <cuda_runtime.h>
#include <dlnne.h>
#include "stdlib.h"
#include <cstring>
// #include "dlcuda/dl_packet_frame.h"
#include "./pluginzoo/NMS/tvm_op/plugin_register.h"

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
    void* data = nullptr;
    bool is_input;
    std::string node_name;
private: 
    Dims shape;
    DataType   dtype;
    int batch_size;
    int index;
    
};

template<typename dataType>
 void read_input_data_to_host(void** input_data_buf, std::string& data_path, int data_len)
{
    std::vector<dataType> numbers;
    dataType number;
    std::ifstream finput;
    if (data_path == "scores_no_path") {
        int tmppp[] = {9, 4, 0, 0, 4, 0 , 0 , 0};
        for(int i = 0; i < 8; i++) {
            for(int j = 0; j < 4096; j++) {
                if(j < tmppp[i]) {
                    numbers.push_back(0.9999f - 0.5 * j / 10);
                } else {
                    numbers.push_back(0.11f);
                }
            }
        } 
    } else {
        finput.open(data_path);
        if (!finput.is_open()) {
            std::cout << "open input data err!" << std::endl;
            return;
        }
        if (data_path == "./data/out1_1_8_4096.txt") { //sorted_idx
            float number_float;
            while (finput >> number_float) {
                numbers.push_back((int)number_float);
            }
        } else {
            while (finput >> number) 
                numbers.push_back(number);
        }

        if (numbers.size() != data_len/sizeof(dataType)) {
            std::cout << "data size err, size = "<< numbers.size() << std::endl;
            return;
        }
        finput.close();
    }

    // 打印读取的数
    // if (data_path == "scores_no_path") { 
    //     for (const auto& num : numbers) {
    //         std::cout << num << std::endl;
    //     }
    // }

    *input_data_buf  = malloc(data_len);
    std::memcpy(*input_data_buf, numbers.data(), data_len);
}



int main(int argc, char **argv)
{
    if(argc < 3) {
        std::cout<<"params err, need specify: model path, batchsize!!"<<std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    int batch_size = atoi(argv[2]);

    // int execute_times = atoi(argv[3]);
    printf("--- init  plugin register \n");
    initPluginRegister();

    int device_id = 0;
    cudaSetDevice(device_id);
    cudaSetClusterMask(1 << 0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int cluster_count = prop.clusterCount;
    auto builder = CreateInferBuilder();
    auto network = builder->CreateNetwork();
    
    BuilderConfig builder_cfg;
    ClusterConfig clusterCfg ;
    // if(2 == cluster_count){
    //     builder_cfg.ws_mode = kShare2;
    //     clusterCfg = kCluster01;
    // }else if (1 == cluster_count){
    //     builder_cfg.ws_mode = kSingle;;
    //     clusterCfg = kCluster0;
    // }else if (4 == cluster_count){
    //     builder_cfg.ws_mode = kShare4;
    //     clusterCfg = kCluster0123;
    // }

    //set to single cluster mode
    builder_cfg.ws_mode = kSingle;;
    clusterCfg = kCluster0;

    builder_cfg.callback = nullptr;
    builder_cfg.max_batch_size = batch_size;
    builder_cfg.dump_dot = true;
    builder_cfg.print_profiling = false;

    int is_weight_shared = 0;
    if( is_weight_shared ) {
        builder->SetBuilderConfig(builder_cfg);
    }

    builder->SetMaxBatchSize(batch_size);
    auto parser = dl::nne::CreateParser();
    auto pwd = std::string(getenv("PWD"));
    std::string front_end_path  = pwd + std::string("/pluginzoo/include/NMS/tvm_op/front_end.py");
    std::string tvm_so_path     = pwd + std::string("/pluginzoo/lib/libDLNms_opt_tvm.so");

    printf("--- register user  op \n");
    parser->RegisterUserOp(tvm_so_path.c_str(), front_end_path.c_str(), "custom_op");

    printf("--- parse model \n");
    parser->Parse(model_path.c_str(),*network);

    printf("--- build engine \n");
    Engine* engine = builder->BuildEngine(*network);

    printf("--- create context \n");
    ExecutionContext* context = nullptr;
    if( is_weight_shared ) {
        context = engine->CreateExecutionContext(clusterCfg);
    } else {
        context = engine->CreateExecutionContext();
    }

    /*read input data to host*/
    // inputs=['origin_boxes','sorted_scores', 'sorted_idx','sort_size'],
    float* origin_boxes_host = nullptr;
    int data_length = 3200 * 4 * sizeof(float);
    std::string data_path = "./data/out0_1_1_4096_4.txt";//boxes
    read_input_data_to_host<float>((void**)&origin_boxes_host,data_path,data_length);

    float* sorted_scores_host = nullptr;
    data_length = 4096 * 8 * sizeof(float);
    data_path = "scores_no_path"; //scores, construct by myself
    read_input_data_to_host<float>((void**)&sorted_scores_host,data_path,data_length);

    int* sorted_idx_host = nullptr;
    data_length = 4096 * 8 * sizeof(int);
    data_path = "./data/out1_1_8_4096.txt"; //idx
    read_input_data_to_host<int>((void**)&sorted_idx_host,data_path,data_length);

    int* sort_size_host = nullptr;
    data_length = 8 * sizeof(int);
    data_path = "./data/out2_1_8.txt"; //sort_size
    read_input_data_to_host<int>((void**)&sort_size_host,data_path,data_length);

    /* create bindings */
    printf("--- create bindings \n");
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

        void* binding_data = nullptr;
        int buf_len = binding->get_batched_node_size();
        {
            CHECK( cudaMalloc(&binding_data, buf_len));
            cudaMemset(binding_data, 0, buf_len);
            binding->data = binding_data;
            binding_device_array[i] = binding_data;
 
            if ( !is_input ) {
                auto phost_int = malloc(buf_len);
                binding_host_array[i] = phost_int;
            }
        }
        
        binding_list.push_back(binding); 
    }

   std::cout << "---- Copy input!" << std::endl;
    /* copy inputdata from host  to device */
    int input_length;
    for(auto i = 0; i < nb_bindings; i++) {
        if( binding_list[i]->is_input ) {
            std::cout << "i == " << i << ", name = " << binding_list[i]->node_name << std::endl;
                if (binding_list[i]->node_name == "origin_boxes") {
                    input_length = binding_list[i]->get_batched_node_size();
                    float* dest = (float*)binding_list[i]->data;
                    if (origin_boxes_host != nullptr) {
                        cudaMemcpy((void*)dest, (void*)origin_boxes_host, input_length, cudaMemcpyHostToDevice);
                    } else {
                        std::cout <<"nullptr err!"<<std::endl;
                    }
                } else if(binding_list[i]->node_name == "sorted_scores") {
                    input_length = binding_list[i]->get_batched_node_size();
                    float* dest = (float*)binding_list[i]->data;
                    if (sorted_scores_host != nullptr) {
                        cudaMemcpy((void*)dest, (void*)sorted_scores_host, input_length, cudaMemcpyHostToDevice);
                    } else {
                        std::cout <<"nullptr err!"<<std::endl;
                    }                    
                } else if(binding_list[i]->node_name == "sorted_idx") {
                    input_length = binding_list[i]->get_batched_node_size();
                    int* dest = (int*)binding_list[i]->data;
                    if (sorted_idx_host != nullptr) {
                        cudaMemcpy((void*)dest, (void*)sorted_idx_host, input_length, cudaMemcpyHostToDevice);
                    } else {
                        std::cout <<"nullptr err!"<<std::endl;
                    }                      
                } else if(binding_list[i]->node_name == "sort_size") {
                    input_length = binding_list[i]->get_batched_node_size();
                    int* dest = (int*)binding_list[i]->data;
                    if (sort_size_host != nullptr) {
                        cudaMemcpy((void*)dest, (void*)sort_size_host, input_length, cudaMemcpyHostToDevice);
                    } else {
                        std::cout <<"nullptr err!"<<std::endl;
                    }                      
                } else {
                    printf("input node err: %s\n",binding_list[i]->node_name.c_str());
                }
        }
    }

    /*do infer*/
    // cudaStream_t m_Stream;
    // cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking);
    // context->Enqueue(batch_size, binding_device_array,m_Stream,nullptr);
    printf("--- start execute\n");
    // for( int i = 0; i < execute_times; i++) {
        context->Execute(batch_size, binding_device_array);
    // }
    
    printf("--- Execute success \n");
    // cudaStreamSynchronize(m_Stream);
    
    // cudaStreamDestroy(m_Stream);
    engine->Destroy();

    int num_detections = 0;
    std::vector<float> nmsed_socres;
    std::vector<float> nmsed_boxes;
     std::vector<int> classes_idx;

    for(auto i = 0; i < nb_bindings; i++) {
        if( !binding_list[i]->is_input ) { 
            int outputdata_len = binding_list[i]->get_batched_node_size();
            std::cout <<binding_list[i]->node_name<<" output_len:"<< outputdata_len <<std::endl;
            cudaMemcpy(binding_host_array[i], binding_list[i]->data, outputdata_len, cudaMemcpyDeviceToHost);

            if(binding_list[i]->node_name == "num_detections") {
                num_detections = ((int*)binding_host_array[i])[0];
            }

            if(binding_list[i]->node_name == "nmsed_scores") {
                for (int j = 0; j < 128; j++) {
                    nmsed_socres.push_back(((float*)binding_host_array[i])[j]);
                }  
            }

            if(binding_list[i]->node_name == "nmsed_boxes") {
                for (int j = 0; j < 128; j++) {
                    nmsed_boxes.push_back(((float*)binding_host_array[i])[j]);
                }  
            }

            if(binding_list[i]->node_name == "classes_idx") {
                for (int j = 0; j < 128; j++) {
                    classes_idx.push_back(((int*)binding_host_array[i])[j]);
                }  
            }                        
        }
    }

    /* print results */
    std::cout<<"-----------------printf result----------------"<<std::endl;
    std::cout<<"num_detections = "<<num_detections<<std::endl;
    for (int i = 0; i < num_detections; i++) {
        std::cout << "class_id = [" << classes_idx[2 * i] << ", " << classes_idx[2 * i + 1] << "]" << ", "
            << "scores = "<< nmsed_socres[i] 

            << ", boxes = [" << nmsed_boxes[4 * i] << ", " << nmsed_boxes[4 * i + 1]
            << ", " << nmsed_boxes[4 * i + 2] << ", " << nmsed_boxes[4 * i + 3] << "]" << std::endl;
    }
    return 0;
}

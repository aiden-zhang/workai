// #include <cuda.h>
#include <dlnne.h>
#include <cuda_runtime_api.h>
#include <cu_ext.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>
#include "utils.h"

using namespace dl::nne;
#define CUDA_CHECK(call) \
    do {\
        cudaError_t _e = call;\
        if (_e != cudaSuccess) {\
            std::cerr << "CUDA error " << _e << ": " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            assert(0);\
        }\
    } while (0)


class Bindings
{
public:
    Bindings(Dims shape, int dataTypeSize, int batch_size, std::string name, bool is_input, int node_size)
    : shape(shape), dataTypeSize(dataTypeSize), batch_size(batch_size), node_name(name), is_input(is_input), node_size(node_size){};

    ~Bindings() = default;

public:
    bool  is_input;
    int   node_size;
    int   dataTypeSize;
    void* host_buffer;
    void* device_buffer;
    std::string node_name;
private: 
    Dims shape;

    int batch_size;
};


size_t getElementSize(dl::nne::DataType dataType) {
    switch (dataType) {
        case dl::nne::kINT8:
        case dl::nne::kUINT8:
            return 1;
        case dl::nne::kINT16:
        case dl::nne::kUINT16:
        case dl::nne::kFLOAT16:
            return 2;
        case dl::nne::kFLOAT32:
        case dl::nne::kINT32:
        case dl::nne::kUINT32:
            return 4;
        case dl::nne::kFLOAT64:
        case dl::nne::kINT64:
        case dl::nne::kUINT64:
            return 8;
        default:
            assert(0);
    }
}



int main(int argc, char *argv[]) {

    #if 1
    int deviceId = 0;
    int maxBatchSize = 32;
    int warmupTimes = 1;
    int executionTimes = 50;
    int threadNum = 1;
    int is_deserialize = 0;
    int is_serialize = 0;
    std::string modelPath = "./justaddfp32.rlym";
    std::vector<std::string> output_nodes;
    getCustomOpt(argc, argv, modelPath, maxBatchSize, executionTimes, threadNum, is_serialize, is_deserialize, output_nodes);

    std::cout << "maxBatchSize: "<< maxBatchSize << ", executionTimes: " << executionTimes <<", threadNum: "<< threadNum 
    << ", is_serialize: "<< is_serialize << ", is_deserialize: "<< is_deserialize << std::endl;

    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

    int clusterCount = deviceProp.clusterCount;
    int tuCount = 0;
    for(int i = 0; i < clusterCount; i++)
    {
        tuCount += deviceProp.tuNum[i];
    }
    std::cout << "tuCount: "<<tuCount<<std::endl;
    std::vector<float> totalMillisecondsList;
    std::vector<int> clusterFpsList;
    for (int i = 0; i < threadNum; ++i) {
        totalMillisecondsList.emplace_back(0);
        clusterFpsList.emplace_back(0);
    }

    std::string serialize_file_name = "./slz.bin"; 
    dl::nne::Engine *engine = nullptr;
    dl::nne::Parser *parser = nullptr;
    dl::nne::Builder *builder =nullptr;
    dl::nne::Network *network = nullptr;

    if (is_deserialize == 1) {
        std::cout << "----now will deserialize---- the serialized engine is: " << serialize_file_name << std::endl;
        std::ifstream slz(serialize_file_name);
        TEST_RETURN_FALSE_EXPR(slz.is_open());

        slz.seekg(0, std::ios::end);
        uint64_t length = static_cast<uint64_t>(slz.tellg());
        slz.seekg(0, std::ios::beg);
        char *slz_data = new char[length];
        slz.read(slz_data, static_cast<int64_t>(length));
        auto start=std::chrono::system_clock::now();
        engine = dl::nne::Deserialize(slz_data, length);
        auto end=std::chrono::system_clock::now();
        std::chrono::duration<double> diff=end-start;
        std::cout<<"deserialize engine cost:"<<diff.count()*1000<<"(ms)"<<std::endl;
        delete[] slz_data;
    } else {

        builder = dl::nne::CreateInferBuilder();
        network = builder->CreateNetwork();
        std::cout << "----SetConfig----\n";
        network->SetConfig("--device-profile=v2.1c.1t.128");

        dl::nne::BuilderConfig builderCfg;
        builderCfg.max_batch_size = maxBatchSize;
        builderCfg.ws_mode = dl::nne::kSingle;

        std::cout << "----SetBuilderConfig----\n";        
        builder->SetBuilderConfig(builderCfg);
        parser = dl::nne::CreateParser();

        // if(custom_pulgin_so_path!="" && custom_pulgin_front_path!=""){
        //     parser->RegisterUserOp(custom_pulgin_so_path.c_str(),custom_pulgin_front_path.c_str(),"custom_op");
        // }
        // for (const auto &input : inputs_dict_) {
        //     parser->RegisterInput(input.first.c_str(), input.second);
        // }

        for(auto iter : output_nodes) {
            parser->RegisterOutput(iter.c_str());
        }

        std::cout << "----Parse model: " << modelPath << "----\n";
        auto parse_ret = parser->Parse(modelPath.c_str(), *network);
        if( parse_ret != 1 ) {
            std::cout << "----parse fail!! ret: "<<parse_ret<<std::endl;
            return -1;  
        }

        std::cout << "----BuildEngine----\n";        
        if( (engine = builder->BuildEngine(*network)) == nullptr ) {
            std::cout << "----build engine fail!!\n";
            return -1;  
        }
    }

    if (is_serialize == 1) {
        std::cout<<"----Serialized---- file saved to "<< serialize_file_name << std::endl;
        auto ser_res = engine->Serialize();
        std::ofstream slz(serialize_file_name);
        TEST_RETURN_FALSE_EXPR(slz.is_open());
        slz.write(static_cast<char *>(ser_res->Data()),
                    static_cast<int64_t>(ser_res->Size()));
        slz.close();
        return 0;
    }

    auto clusterEnqueueFunc = [&](int batch_size, int clusterIdx, int thread_idx) -> int {
        std::cout << "clusterIdx: " << clusterIdx << " --> threadIdx: "<< thread_idx << std::endl;
        auto nbBindings = engine->GetNbBindings();
        void **binding_device_array = (void**)malloc(nbBindings * sizeof(void*));
        // void *deviceBuffers[nbBindings];
        std::vector<Bindings*> binding_list;

        for (int i = 0; i < nbBindings; ++i) {
            auto dataType     = engine->GetBindingDataType(i);
            auto name = engine->GetBindingName(i);
            auto dims = engine->GetBindingDimensions(i);
            auto data_type = engine->GetBindingDataType(i);
            bool is_input = engine->BindingIsInput(i);

            auto elementCount = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
            auto dataTypeSize = getElementSize(dataType);
            auto bindingSize  = batch_size*elementCount*dataTypeSize;
            auto binding = new Bindings(dims, dataTypeSize, batch_size, name, is_input, bindingSize);

            void* pdev = nullptr;
            void* phost = nullptr;
            CUDA_CHECK( cudaMalloc(&pdev, bindingSize));
            phost = malloc(bindingSize);

            binding->device_buffer = pdev;
            binding->host_buffer = phost;
            
            printf("dataTypeSize: %zu, bindingSize: %zu\n", dataTypeSize,bindingSize);
            if (is_input) {
                for(int i = 0; i < bindingSize/dataTypeSize ; i++)
                {
                    if(dataTypeSize == 1) {
                        ((char*)phost)[i] = 1;
                    } else if (dataTypeSize == 4) 
                    {
                        ((float*)phost)[i] = 1.0f;
                    } else {
                        printf("err! dataTypeSize: %zu\n", dataTypeSize);
                        return -1;
                    }
                }
                cudaMemcpy(pdev, phost, bindingSize, cudaMemcpyHostToDevice);
            }       

            binding_list.push_back(binding); 
            /* copy input */

            binding_device_array[i] = pdev;
        }

        auto clusterConfig = static_cast<dl::nne::ClusterConfig>(clusterIdx);
        auto executionContext = engine->CreateExecutionContext(clusterConfig);
        if( executionContext == nullptr ) {
            std::cout << "create context fail!!\n";
            return -1;  
        }
        cudaStream_t cudaStream;
        CUDA_CHECK( cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking) );
        float &totalMilliseconds = totalMillisecondsList[thread_idx];    
        for (int i = 0; i < warmupTimes; ++i) {
            executionContext->Enqueue(maxBatchSize, binding_device_array, cudaStream, nullptr);
            CUDA_CHECK( cudaStreamSynchronize(cudaStream) );
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < executionTimes; ++i) {
            executionContext->Enqueue(maxBatchSize, binding_device_array, cudaStream, nullptr);
            CUDA_CHECK( cudaStreamSynchronize(cudaStream) );
        }
        auto end = std::chrono::high_resolution_clock::now();        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalMilliseconds = duration;

        /* copy output */
        for(auto i = 0; i < nbBindings; i++) {
            if( !binding_list[i]->is_input ) { //only test one output node
                std::cout << "output_len:"<< binding_list[i]->node_size << " node_name" << binding_list[i]->node_name <<std::endl;
                cudaMemcpy(binding_list[i]->host_buffer, binding_list[i]->device_buffer, binding_list[i]->node_size, cudaMemcpyDeviceToHost);
                for (int kk = 0; kk < batch_size; kk++) {
                    for( int j = 0; j < 3; j++)
                    {
                        if(binding_list[i]->dataTypeSize == 1) {
                            printf("%d  ", ((char*)binding_list[i]->host_buffer)[kk*10+j]);
                        } else if (binding_list[i]->dataTypeSize == 4) {
                            printf("%f  ", ((float*)binding_list[i]->host_buffer)[kk*10+j]);
                        } else {
                            printf("err! dataTypeSize: %d\n", binding_list[i]->dataTypeSize);
                            return -1;
                        }
                    }
                }
            }
        }

        /* release resource */
        for (int i = 0; i < nbBindings; ++i) {
            CUDA_CHECK( cudaFree(binding_device_array[i]) );
        }

        CUDA_CHECK( cudaStreamDestroy(cudaStream) );
        executionContext->Destroy();

        return 0;
    };


    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; ++i) {
        std::cout<< "theead: " << i <<" @ tu: " << i % tuCount << std::endl;
        auto thread = std::thread(clusterEnqueueFunc, maxBatchSize, i % tuCount, i);
        threads.emplace_back(std::move(thread));
    }

    for (int i = 0; i < threadNum; ++i) {
        threads[i].join();
    }

    float SumMilliseconds = 0;
    int allClusterFps = 0;
    for (int i = 0; i < threadNum; ++i) {
        clusterFpsList[i] = 1000.0f * maxBatchSize *executionTimes/totalMillisecondsList[i];
        std::cout << "\nthread " << i << "; costs: " << totalMillisecondsList[i] / executionTimes << " ms; fps: "<< clusterFpsList[i] << std::endl;
        SumMilliseconds+=totalMillisecondsList[i];
        allClusterFps += clusterFpsList[i];
    }
    
    std::cout << "avg cost: " << SumMilliseconds/ threadNum/executionTimes<< " ms, all fps: " << allClusterFps << std::endl;
    
    // engine->Destroy();
    if (is_deserialize != 1) {
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
    }
    #endif
    // exit(0);
    return 0;
}
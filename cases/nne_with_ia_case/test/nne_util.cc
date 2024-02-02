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
#include <mutex>
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



int nne_run(void* inputFrame, int inputSize, cudaStream_t cudaStream) {

    #if 1
    int deviceId = 0;
    int maxBatchSize = 1;
    int warmupTimes = 0;
    int executionTimes = 50;
    int threadNum = 2;

    std::string modelPath = "../multiply_divide_int8.rlym";
    std::vector<std::string> output_nodes;

    std::cout << "modelPath: "<< modelPath << ", maxBatchSize: "<< maxBatchSize << ", executionTimes: " << executionTimes <<", threadNum: "<< threadNum 
    << std::endl;

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

    // serialize_file_name = modelPath;   
    dl::nne::Engine *engine = nullptr;
    dl::nne::Parser *parser = nullptr;
    dl::nne::Builder *builder =nullptr;
    dl::nne::Network *network = nullptr;
    dl::nne::ErrorRecorder *errorcode = (ErrorRecorder*)malloc(sizeof(ErrorRecorder));
    std::vector<ExecutionContext*> contextList;



    builder = dl::nne::CreateInferBuilder();
    network = builder->CreateNetwork();
    std::cout << "----SetConfig----\n";
    auto tmp = network->SetConfig("--device-profile=v2.1c.1t.128");
    // printf("GetErrorDesc: %d\n",tmp);
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
    // std::string output_nodes;
    // for (auto node : output_nodes_) {
    //     TEST_RETURN_FALSE_EXPR(parser->RegisterOutput(node.c_str()));
    // }

    for(auto iter : output_nodes) {
        parser->RegisterOutput(iter.c_str());
    }

    std::cout << "----Parse----\n";
    parser->Parse(modelPath.c_str(), *network);

    std::cout << "----BuildEngine----\n";        
    engine = builder->BuildEngine(*network);

    for (int i = 0; i < threadNum; ++i) {
        totalMillisecondsList.emplace_back(0);
        clusterFpsList.emplace_back(0);
        auto clusterConfig = static_cast<dl::nne::ClusterConfig>(i);
        contextList.push_back(engine->CreateExecutionContext(clusterConfig));
    }


    // if (is_serialize == 1 && is_deserialize !=1) {
    //     std::cout<<"----Serialized---- file saved to "<< serialize_file_name << std::endl;
    //     auto ser_res = engine->Serialize();
    //     std::ofstream slz(serialize_file_name);
    //     TEST_RETURN_FALSE_EXPR(slz.is_open());
    //     slz.write(static_cast<char *>(ser_res->Data()),
    //                 static_cast<int64_t>(ser_res->Size()));
    //     slz.close();
    // }

    std::mutex m_mutex;
    auto clusterEnqueueFunc = [&](int batch_size, int clusterIdx, int thread_idx) -> int {
        std::lock_guard<std::mutex> tmp_mutext(m_mutex); 
        std::cout << "clusterIdx: " << clusterIdx << " --> threadIdx: "<< thread_idx << std::endl;
        
        /* 1. create binding buffer */
        auto nbBindings = engine->GetNbBindings();
        void **binding_device_array = (void**)malloc(nbBindings * sizeof(void*));
        // void *deviceBuffers[nbBindings];
        std::vector<Bindings*> binding_list;
        
        for (int i = 0; i < nbBindings; ++i) {
            auto dataType  = engine->GetBindingDataType(i);
            auto name      = engine->GetBindingName(i);
            auto dims      = engine->GetBindingDimensions(i);
            auto data_type = engine->GetBindingDataType(i);
            bool is_input  = engine->BindingIsInput(i);

            auto elementCount = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
            auto dataTypeSize = getElementSize(dataType);
            auto bindingSize  = batch_size*elementCount*dataTypeSize;
            auto binding = new Bindings(dims, dataTypeSize, batch_size, name, is_input, bindingSize);

            printf("---name: %s, dataTypeSize: %zu, bindingSize: %lu, inputSize: %d\n", name, dataTypeSize, bindingSize, inputSize);
            if(inputSize*batch_size != bindingSize) {
                printf("bindingSize err!");
                return -1;
            }
            
            void* pdev = nullptr;
            void* phost = nullptr;
            CUDA_CHECK(cudaMalloc(&pdev, bindingSize));
            phost = malloc(bindingSize);

            binding->device_buffer = pdev;
            binding->host_buffer = phost;

            // if (is_input) {
            //     for(int i = 0; i < bindingSize/dataTypeSize ; i++)
            //     {
            //         if(dataTypeSize == 1) {
            //             ((char*)phost)[i] = 1;
            //         } else if (dataTypeSize == 4) 
            //         {
            //             ((float*)phost)[i] = 1.0f;
            //         } else {
            //             printf("err! dataTypeSize: %zu\n", dataTypeSize);
            //             return;
            //         }
            //     }
            //     cudaMemcpy(pdev, phost, bindingSize, cudaMemcpyHostToDevice);
            // }       
            
            
            /* copy input */
            for (int k  = 0; k <  batch_size; k++) {
                //input is uint8
                cudaMemcpy((char*)pdev+k*inputSize, inputFrame, inputSize, cudaMemcpyDeviceToDevice);
            }
            binding_list.push_back(binding); 
            binding_device_array[i] = pdev;
        }


        /* 2. do infer */
        // cudaStream_t cudaStream;
        // CUDA_CHECK( cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking) );
        float &totalMilliseconds = totalMillisecondsList[thread_idx];    

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < executionTimes; ++i) {
            contextList[thread_idx]->Enqueue(maxBatchSize, binding_device_array, cudaStream, nullptr);
            CUDA_CHECK( cudaStreamSynchronize(cudaStream) );
        }

        auto end = std::chrono::high_resolution_clock::now();        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalMilliseconds = duration;
   

        /* 3. copy output */
        #if 1
        std::vector<int> golden = {82, 82, -18, 34};

        for(auto i = 0; i < nbBindings; i++) {
            if( !binding_list[i]->is_input ) {   //only test one output node
                std::cout << "---output_len: "<< binding_list[i]->node_size << " node_name: " << binding_list[i]->node_name <<std::endl;
                auto bindingSize = binding_list[i]->node_size;
                cudaMemcpy(binding_list[i]->host_buffer, binding_list[i]->device_buffer, bindingSize, cudaMemcpyDeviceToHost);

                printf("---gpuoutput: ");
                for (int bb = 0; bb < batch_size; bb++) {
                    for( int j = 0; j < 4; j++)
                    {
                        if(binding_list[i]->dataTypeSize == 1) {
                            auto outputgpu = ((char*)binding_list[i]->host_buffer)[bb*bindingSize/batch_size+1000*j];
                            printf("%d  ", outputgpu);
                            if(outputgpu != golden[j]) {
                                printf("---matched fail!! gpu output: %d, cpu golden: %d\n", outputgpu, golden[j]);
                                return -1;
                            }

                        } else if (binding_list[i]->dataTypeSize == 4) {
                            // printf("%f  ", ((float*)binding_list[i]->host_buffer)[bb*bindingSize/batch_size+1000*j]);
                        } else {
                            printf("err! dataTypeSize: %d\n", binding_list[i]->dataTypeSize);
                            return -1;
                        }
                    }
                }
                printf("---\n");
            }
        }
        #endif
        // /* release resource */
        for (int i = 0; i < nbBindings; ++i) {
            CUDA_CHECK( cudaFree(binding_device_array[i]) );
        }

        // CUDA_CHECK( cudaStreamDestroy(cudaStream) );
        // executionContext->Destroy();
        // return 1;
        std::cout <<"---thread exit"<<std::endl;
        return 1;
    };




    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; ++i) {
        std::cout<< "thread: " << i <<" @ tu: " << i % tuCount << std::endl;
        auto thread = std::thread(clusterEnqueueFunc, maxBatchSize, i % tuCount, i);
        threads.emplace_back(std::move(thread));
    }

    for (int i = 0; i < threadNum; ++i) {
        threads[i].join();
    }

    // float SumMilliseconds = 0;
    // int allClusterFps = 0;
    // for (int i = 0; i < threadNum; ++i) {
    //     clusterFpsList[i] = 1000.0f * maxBatchSize *executionTimes/totalMillisecondsList[i];
    //     std::cout << "thread " << i << "; costs: " << totalMillisecondsList[i] / executionTimes << " ms; fps: "<< clusterFpsList[i] << std::endl;
    //     SumMilliseconds+=totalMillisecondsList[i];
    //     allClusterFps += clusterFpsList[i];
    // }
    
    // std::cout << "avg cost: " << SumMilliseconds/ threadNum/executionTimes<< " ms, all fps: " << allClusterFps << std::endl;
    
    // engine->Destroy();

    // parser->Destroy();
    // network->Destroy();
    // builder->Destroy();

    #endif
    return 0;
}

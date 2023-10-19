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

#define CUDA_CHECK(call) \
    do {\
        cudaError_t _e = call;\
        if (_e != cudaSuccess) {\
            std::cerr << "CUDA error " << _e << ": " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            assert(0);\
        }\
    } while (0)


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
    int deviceId = 0;
    int maxBatchSize = 32;
    int warmupTimes = 1;
    int executionTimes = 50;
    int threadNum = 4;
    int is_deserialize = 1;
    int is_serialize = 0;
    std::string modelPath = "./humdet.rlym";
    std::vector<std::string> output_nodes;
    getCustomOpt(argc, argv, modelPath, maxBatchSize, executionTimes, threadNum, is_serialize, is_deserialize, output_nodes);

    std::cout << "modelPath: "<< modelPath << ", maxBatchSize: "<< maxBatchSize << ", executionTimes: " << executionTimes <<", threadNum: "<< threadNum 
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
        std::cout << "----now deserialize---- the serialized engine is: " << serialize_file_name << std::endl;
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
        for(auto iter : output_nodes) {
            parser->RegisterOutput(iter.c_str());
        }
        std::cout << "----Parse----\n";
        parser->Parse(modelPath.c_str(), *network);

        std::cout << "----BuildEngine----\n";        
        engine = builder->BuildEngine(*network);
    }


    if (is_serialize == 1 && is_deserialize !=1) {
        std::cout<<"----Serialized---- file saved to "<< serialize_file_name << std::endl;
        auto ser_res = engine->Serialize();
        std::ofstream slz(serialize_file_name);
        TEST_RETURN_FALSE_EXPR(slz.is_open());
        slz.write(static_cast<char *>(ser_res->Data()),
                    static_cast<int64_t>(ser_res->Size()));
        slz.close();
    }

    auto clusterEnqueueFunc = [&](int clusterIdx, int thread_idx) {
        std::cout << "clusterIdx: " << clusterIdx << " --> threadIdx: "<< thread_idx << std::endl;
        auto nbBindings = engine->GetNbBindings();
        void *deviceBuffers[nbBindings];
        for (int i = 0; i < nbBindings; ++i) {
            auto dims         = engine->GetBindingDimensions(i);
            auto dataType     = engine->GetBindingDataType(i);
            auto elementCount = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
            auto dataTypeSize = getElementSize(dataType);
            auto bindingSize  = elementCount * dataTypeSize;

            void *deviceBuffer;
            CUDA_CHECK( cudaMalloc(&deviceBuffer, bindingSize * maxBatchSize) );
            deviceBuffers[i] = deviceBuffer;
        }

        auto clusterConfig = static_cast<dl::nne::ClusterConfig>(clusterIdx);
        auto executionContext = engine->CreateExecutionContext(clusterConfig);

        cudaStream_t cudaStream;
        CUDA_CHECK( cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking) );
        float &totalMilliseconds = totalMillisecondsList[thread_idx];    
        for (int i = 0; i < warmupTimes; ++i) {
            executionContext->Enqueue(maxBatchSize, deviceBuffers, cudaStream, nullptr);
            CUDA_CHECK( cudaStreamSynchronize(cudaStream) );
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < executionTimes; ++i) {
            executionContext->Enqueue(maxBatchSize, deviceBuffers, cudaStream, nullptr);
            CUDA_CHECK( cudaStreamSynchronize(cudaStream) );
        }
        auto end = std::chrono::high_resolution_clock::now();        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalMilliseconds = duration;

        for (int i = 0; i < nbBindings; ++i) {
            CUDA_CHECK( cudaFree(deviceBuffers[i]) );
        }

        CUDA_CHECK( cudaStreamDestroy(cudaStream) );
        executionContext->Destroy();
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; ++i) {
        std::cout<< "theead: " << i <<" @ tu: " << i % tuCount << std::endl;
        auto thread = std::thread(clusterEnqueueFunc, i % tuCount, i);
        threads.emplace_back(std::move(thread));
    }

    for (int i = 0; i < threadNum; ++i) {
        threads[i].join();
    }

    float SumMilliseconds = 0;
    int allClusterFps = 0;
    for (int i = 0; i < threadNum; ++i) {
        clusterFpsList[i] = 1000.0f * maxBatchSize *executionTimes/totalMillisecondsList[i];
        std::cout << "thread " << i << "; costs: " << totalMillisecondsList[i] << " ms; fps: "<< clusterFpsList[i] << std::endl;
        SumMilliseconds+=totalMillisecondsList[i];
        allClusterFps += clusterFpsList[i];
    }
    
    std::cout << "avg cost: " << SumMilliseconds/threadNum << " ms, all fps: " << allClusterFps << std::endl;
    
    engine->Destroy();
    if (is_deserialize != 1) {
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
    }

    return 0;
}








#include <cmath>
#include <mutex>
#include <thread>
#include <cu_ext.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <set>
#include "utils/dl_utils.h"
#include "gtest/gtest.h"

#include "topk/topk_interface.h"
#include "topk/topk_select/topk_select.h"

extern int queryTopkWorkSpaceSize(int k, int count, int input_buffer_num, bool isHalf);
extern void topk16(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk16(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);

extern void topk32(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk64(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk128(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk256(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk512(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk1024(float* input, float* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);

extern void topk32(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk64(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk128(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk256(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk512(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);
extern void topk1024(half* input, half* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream);

extern void topk10000(unsigned char* input, int* output_index, void* work_space,int tile_size,cudaStream_t& stream);
extern int queryTopkWorkSpaceSize(int tile_size);


template<int K,typename DataType>
void topk_sample(DataType* input, DataType* output, int* output_index, void* work_space, int tile_size, cudaStream_t& stream){
	switch(K){
		case 16:topk16(input, output, output_index, work_space, tile_size, stream);break;
		case 32:topk32(input, output, output_index, work_space, tile_size, stream);break;
		case 64:topk64(input, output, output_index, work_space, tile_size, stream);break;
		case 128:topk128(input, output, output_index, work_space, tile_size, stream);break;
		case 256:topk256(input, output, output_index, work_space, tile_size, stream);break;
		case 512:topk512(input, output, output_index, work_space, tile_size, stream);break;
		case 1024:topk1024(input, output, output_index, work_space, tile_size, stream);break;
	}
}

std::mutex g_topk_mutex;
extern int g_cluster_count;

namespace AETest {

static const int TEST_LOOP_COUNT = 16;

template<typename DataType, int K, bool isHalf = false>
static void setupInput(DataType* input, int in_size, DataType* topk_value, int* topk_index) {
    DataType *host_input = (DataType*)malloc(in_size * sizeof(DataType));
    for(int i = 0; i < in_size; i++) {
        host_input[i] = i * 3 % 100 * 1.0f;
    }

    g_topk_mutex.lock();
    srand(time(0));
    for(int i = 0; i < K; i++) {
        topk_index[i] = rand() % in_size;
        bool isRepeated = false;
        for(int j = 0; j < i; j++) {
            if(topk_index[j] == topk_index[i]) {
                isRepeated = true;
            }
        }
        if(isRepeated) {
            i--;
            continue;
        }
        topk_value[i] = 100.0f + K - i;
        host_input[topk_index[i]] = topk_value[i];
    }
    g_topk_mutex.unlock();

    cudaMemcpy(input, host_input, in_size * sizeof(DataType), cudaMemcpyHostToDevice);
    free(host_input);
}

template<typename DataType, int K, bool isHalf = false>
static bool checkResult(DataType* output_value, int* output_index, DataType* topk_value, int* topk_index) {
    DataType* host_output_value = (DataType*)malloc(K * sizeof(DataType));
    int* host_output_index = (int*)malloc(K * sizeof(int));
    cudaMemcpy(host_output_value, output_value, K * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_output_index, output_index, K * sizeof(int), cudaMemcpyDeviceToHost);

    bool ret = true;
    std::stringstream s_topk_index;
    std::stringstream s_cal_index;
    std::stringstream s_topk_value;
    std::stringstream s_cal_value;
    for(int i = 0; i < K; ++i) {
        s_topk_index << topk_index[i] << " ";
        s_cal_index << host_output_index[i] << " ";
        if(host_output_index[i] != topk_index[i]) {
            ret = false;
        }

        float topk_value_tmp = 0.0f;
        float host_output_value_tmp = 0.0f;
        if(isHalf) {
            topk_value_tmp = __half2float(topk_value[i]);
            host_output_value_tmp = __half2float(host_output_value[i]);
        } else {
            topk_value_tmp = topk_value[i];
            host_output_value_tmp = host_output_value[i];
        }
        s_topk_value << topk_value_tmp << " ";
        s_cal_value << host_output_value_tmp << " ";
        if(fabs(host_output_value_tmp - topk_value_tmp) >= 0.0001) {
            ret = false;
        }
    }
    if(!ret) {
        DlLogE << "Src index = " << s_topk_index.str();
        DlLogE << "Dst index = " << s_cal_index.str();
        DlLogE << "Src value = " << s_topk_value.str();
        DlLogE << "Dst value = " << s_cal_value.str();
    }
    free(host_output_value);
    free(host_output_index);

    return ret;
}

class TopkTest : public testing::Test
{
public:
    template<typename DataType, int K, bool isHalf = false>
    bool checkFunction(int in_size, int cluster_count = 4) {
        DataType** d_input_ptr = (DataType**)malloc(sizeof(DataType*) * cluster_count);
        DataType** d_output = (DataType**)malloc(sizeof(DataType*) * cluster_count);
        int** d_output_index = (int**)malloc(sizeof(int*) * cluster_count);
        void** d_tempo_ptr = (void**)malloc(sizeof(void*) * cluster_count);

        int topk_index[4][K];
        DataType topk_value[4][K];
        for(int i = 0; i < cluster_count; ++i)
        {
            cudaSetClusterMask(1 << i);
            cudaMalloc((void**)(d_input_ptr + i), sizeof(DataType) * in_size);
            setupInput<DataType, K, isHalf>(d_input_ptr[i], in_size, topk_value[i], topk_index[i]);

            cudaMalloc((void**)(d_output + i), sizeof(DataType) * K);
            cudaMalloc((void**)(d_output_index + i), sizeof(int) * K);

            int temp_size = queryTopkWorkSpaceSize(K, in_size, 1, isHalf);
            cudaMalloc((void**)(d_tempo_ptr + i), temp_size);
        }

        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);
        for(int m = 0; m < cluster_count; ++m) {
            cudaSetClusterMask(1 << m);
            topk_sample<K>(d_input_ptr[m], d_output[m], d_output_index[m], d_tempo_ptr[m], in_size, stream);
        }
        cudaStreamSynchronize(stream);

        bool res = true;
        for(int i = 0; i < cluster_count; ++i) {
            res &= checkResult<DataType, K, isHalf>(d_output[i], d_output_index[i], topk_value[i], topk_index[i]);
        }

        for(int i = 0; i < cluster_count; ++i) {
            cudaFree(d_tempo_ptr[i]);
            cudaFree(d_output_index[i]);
            cudaFree(d_output[i]);
            cudaFree(d_input_ptr[i]);
        }
        cudaStreamDestroy(stream);
        free(d_tempo_ptr);
        free(d_output_index);
        free(d_output);
        free(d_input_ptr);

        return res;
    }

    template<typename DataType, int K, bool isHalf = false>
    float checkPerformance(int in_size, int cluster_count = 4) {
        DataType** d_input_ptr = (DataType**)malloc(sizeof(DataType*) * cluster_count);
        DataType** d_output = (DataType**)malloc(sizeof(DataType*) * cluster_count);
        int** d_output_index = (int**)malloc(sizeof(int*) * cluster_count);
        void** d_tempo_ptr = (void**)malloc(sizeof(void*) * cluster_count);

        int topk_index[4][K];
        DataType topk_value[4][K];
        for(int i = 0; i < cluster_count; ++i)
        {
            cudaSetClusterMask(1 << i);
            cudaMalloc((void**)(d_input_ptr + i), sizeof(DataType) * in_size);
            setupInput<DataType, K, isHalf>(d_input_ptr[i], in_size, topk_value[i], topk_index[i]);

            cudaMalloc((void**)(d_output + i), sizeof(DataType) * K);
            cudaMalloc((void**)(d_output_index + i), sizeof(int) * K);

            int temp_size = queryTopkWorkSpaceSize(K, in_size, 1, isHalf);
            cudaMalloc((void**)(d_tempo_ptr + i), temp_size);
        }

        int warmup_times = 10;
        int logout_times = 10;
        int excute_times = 50;
        float performance = 0.0f;
        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);
        auto dl_timer = DlTimerFactory::getInstance().createDlTimer(DlTimerType_CPU);
        for(int excute_time = 0; excute_time < excute_times; ++excute_time) {
            if(excute_time >= warmup_times) {
                dl_timer->start();
            }

            DataType threshold = 127.5f;
            for(int m = 0; m < cluster_count; ++m) {
                cudaSetClusterMask(1 << m);
                topk_sample<K>(d_input_ptr[m], d_output[m], d_output_index[m], d_tempo_ptr[m], in_size, stream);
            }

            if(excute_time < warmup_times || 0 == (excute_time - warmup_times + 1) % logout_times) {
                cudaStreamSynchronize(stream);
            }
            if(excute_time >= warmup_times) {
                dl_timer->stop();
                if(0 == dl_timer->total_count() % logout_times) {
                    performance = dl_timer->total_elapsed() / dl_timer->total_count();
                    DlLogD << "It takes average " << dl_timer->total_elapsed() / dl_timer->total_count()
                           << " ms to excute mark " << dl_timer->total_count() << " times!";
                    dl_timer->reset();
                }
            }
        }
        DlTimerFactory::getInstance().releaseDlTimer(dl_timer);
        cudaStreamSynchronize(stream);

        for(int i = 0; i < cluster_count; ++i) {
            cudaFree(d_tempo_ptr[i]);
            cudaFree(d_output_index[i]);
            cudaFree(d_output[i]);
            cudaFree(d_input_ptr[i]);
        }
        cudaStreamDestroy(stream);
        free(d_tempo_ptr);
        free(d_output_index);
        free(d_output);
        free(d_input_ptr);

        return performance;
    }
};

class TopkSelectTest : public testing::Test
{
public:
    std::set<int> fake_data(unsigned char* input_data, int size, int K = 10000) {
            srand(1);
            int pre_bin_num=32;
            for (int i = 0; i < size; i++) {
                unsigned int value = pre_bin_num + (rand() % (256 - pre_bin_num));
                input_data[i] = (unsigned char) value; //put small value on prefix 31 bin,only for test



            }
            std::set<int> address;
            

            //write conner case
            for(int i=0;i<5;i++){
                input_data[size-1-i]=(unsigned int) (rand() % pre_bin_num);
                address.insert(size-1-i);
            }
            int k = 5;
            do {
                int ad = rand() % (size-5);
                if (address.find(ad) == address.end()) {
                    address.insert(ad);
                    input_data[ad] = (unsigned int) (rand() % pre_bin_num);
                    k++;
                }
            } while (k < K);
            return address;
    }

    bool checkResult(std::set<int> golden_idxs,int* result,int K){
        bool success=true;
        for (int i = 0; i < K; i++) {
            int address=result[i];
            if(golden_idxs.find(address)==golden_idxs.end()){
                std::cout <<"fail:"<<i<<"th\t" << address<<" not found in golden ......"<<std::endl;
                success=false;
                break;
            }
        }
        //if(success){
        //    std::cout <<"check:Pass..."<<std::endl;
        //}
        return success;
    }
    template<typename DataType, int K> //DataType must be uchar,K =10000
    bool checkFunction(int in_size, int cluster_count = 4) {
        DataType** d_input_ptr = (DataType**)malloc(sizeof(DataType*) * cluster_count);
        
        DataType** cpu_input_ptr = (DataType**)malloc(sizeof(DataType*) * cluster_count);

     
        int** d_output_index = (int**)malloc(sizeof(int*) * cluster_count);
       
        void** d_tempo_ptr = (void**)malloc(sizeof(void*) * cluster_count);

        int** topk_index= (int**)malloc(sizeof(int*) * cluster_count);
        
        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);

        std::set<int> golden_idxs[4];
        
        for(int i = 0; i < cluster_count; ++i)
        {
            topk_index[i]=(int*)malloc(sizeof(int)*K);
            cpu_input_ptr[i]=(unsigned char* )malloc(sizeof(int)*in_size);

            cudaSetClusterMask(1 << i);
            cudaMalloc((void**)(d_input_ptr + i), sizeof(DataType) * in_size);

            golden_idxs[i]=fake_data(cpu_input_ptr[i],in_size,K);
            

            cudaMemcpyAsync(d_input_ptr[i],cpu_input_ptr[i],sizeof(DataType)*in_size,cudaMemcpyHostToDevice,stream);

            //cudaMalloc((void**)(d_output + i), sizeof(DataType) * K);
            cudaMalloc((void**)(d_output_index + i), sizeof(int) * K);

            int temp_size = queryTopkWorkSpaceSize(in_size); //query uchar top10000

            
            cudaMalloc((void**)(d_tempo_ptr + i), temp_size);
        }

        
        for(int m = 0; m < cluster_count; ++m) {
            cudaSetClusterMask(1 << m);
            topk10000(d_input_ptr[m],d_output_index[m], d_tempo_ptr[m], in_size, stream);
            cudaMemcpyAsync(topk_index[m],d_output_index[m],sizeof(int)*K,cudaMemcpyDeviceToHost,stream);
        }
        
        cudaStreamSynchronize(stream);

        bool res = true;
        
        for(int i = 0; i < cluster_count; ++i) {
            res &= checkResult(golden_idxs[i], topk_index[i],K);
        }

        for(int i = 0; i < cluster_count; ++i) {
            cudaFree(d_tempo_ptr[i]);
            cudaFree(d_output_index[i]);
            //cudaFree(d_output[i]);
            cudaFree(d_input_ptr[i]);

            free(cpu_input_ptr[i]);
            free(topk_index[i]);
        }
        cudaStreamDestroy(stream);
        free(d_tempo_ptr);
        free(d_output_index);
        free(topk_index);
        free(d_input_ptr);

        return res;
    }

    template<typename DataType, int K> //DataType must be uchar,K =10000
    float checkPerformance(int in_size, int cluster_count = 4) {
        DataType** d_input_ptr = (DataType**)malloc(sizeof(DataType*) * cluster_count);
        
        DataType** cpu_input_ptr = (DataType**)malloc(sizeof(DataType*) * cluster_count);

     
        int** d_output_index = (int**)malloc(sizeof(int*) * cluster_count);
       
        void** d_tempo_ptr = (void**)malloc(sizeof(void*) * cluster_count);

        int** topk_index= (int**)malloc(sizeof(int*) * cluster_count);
        
        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);

        std::set<int> golden_idxs[4];

        for(int i = 0; i < cluster_count; ++i)
        {
            topk_index[i]=(int*)malloc(sizeof(int)*K);

            cpu_input_ptr[i]=(unsigned char* )malloc(sizeof(int)*in_size);
            
            cudaSetClusterMask(1 << i);
            cudaMalloc((void**)(d_input_ptr + i), sizeof(DataType) * in_size);
            golden_idxs[i]=fake_data(cpu_input_ptr[i],in_size,K);
            cudaMemcpyAsync(d_input_ptr[i],cpu_input_ptr[i],sizeof(DataType)*in_size,cudaMemcpyHostToDevice,stream);

        
            cudaMalloc((void**)(d_output_index + i), sizeof(int) * K);

            int temp_size = queryTopkWorkSpaceSize(in_size); //query uchar top10000
            cudaMalloc((void**)(d_tempo_ptr + i), temp_size);
        }
        cudaStreamSynchronize(stream);
        auto dl_timer = DlTimerFactory::getInstance().createDlTimer(DlTimerType_CPU);
        dl_timer->start();
        for(int i=0;i<100;i++){
            for(int m = 0; m < cluster_count; ++m) {
                cudaSetClusterMask(1 << m);
                topk10000(d_input_ptr[m],d_output_index[m], d_tempo_ptr[m], in_size, stream);
            }
        }

        for(int m = 0; m < cluster_count; ++m) {
            cudaMemcpyAsync(topk_index[m],d_output_index[m],sizeof(int)*K,cudaMemcpyDeviceToHost,stream);
        }    
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        dl_timer->stop();

        float performance = dl_timer->total_elapsed() / 100;

        for(int i = 0; i < cluster_count; ++i) {
            cudaFree(d_tempo_ptr[i]);
            cudaFree(d_output_index[i]);
            //cudaFree(d_output[i]);
            cudaFree(d_input_ptr[i]);

            free(cpu_input_ptr[i]);
            free(topk_index[i]);
        }
        cudaStreamDestroy(stream);
        free(d_tempo_ptr);
        free(d_output_index);
        free(topk_index);
        free(d_input_ptr);

        return performance;
    }
};

TEST_F(TopkSelectTest,Function)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024+16*4, 128*1024*1024};
    std::string test_name = "<AETest::TopkSelectTest::Function";
    try {
        constexpr int K = 10000;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<unsigned char, K>(params.at(i) / g_cluster_count, g_cluster_count);
              
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}
TEST_F(TopkSelectTest,Performance)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024+16*4, 128*1024*1024};
    std::vector<float> golden = {0.62f, 1.1f, 5.1f};

    std::string test_name = "<AETest::TopkSelectTest::Performance";
    {
        constexpr int K = 10000;
       
        for(unsigned int i = 0; i < params.size(); ++i) {
            
            auto performance= checkPerformance<unsigned char, K>(params.at(i)/ g_cluster_count, g_cluster_count);
            
            std::cout << test_name << "@"<<params.at(i)<<":"<< performance<<"(ms)" << std::endl;
            
            if(golden.at(i)<=performance){
               std::cout <<"Fail perofmance regression >="<<golden.at(i) << std::endl;
            }
        }
    }
}


TEST_F(TopkTest, Function)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 64;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}


TEST_F(TopkTest, ThreadsFunction)
{
    std::vector<std::pair<int, int>> params;
    params.emplace_back(std::pair<int, int>(8*1024*1024, 2));
    params.emplace_back(std::pair<int, int>(8*1024*1024, 4));
    params.emplace_back(std::pair<int, int>(8*1024*1024, 8));
    params.emplace_back(std::pair<int, int>(8*1024*1024, 16));
    params.emplace_back(std::pair<int, int>(8*1024*1024, 32));

    params.emplace_back(std::pair<int, int>(4*1024*1024, 4));
    params.emplace_back(std::pair<int, int>(2*1024*1024, 8));
    params.emplace_back(std::pair<int, int>(1*1024*1024, 16));

    std::string test_name = "<AETest::TopkTest::ThreadsFunction";
    try {
        const int K = 64;
        bool isPass = true;
        for(unsigned int index = 0; index < params.size(); ++index) {
            auto itr = params.at(index);
            int size = itr.first;
            int thread_count = itr.second;
            std::thread** thread = (std::thread**)malloc(thread_count * sizeof(std::thread*));
            static const int MAX_THREAD_COUNT = 128;
            bool thread_result[MAX_THREAD_COUNT] = {0};
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                for(int i = 0; i < thread_count; i++) {
                    thread[i] = new std::thread([this](int size, int tid, bool *result)->void {
                                                result[tid] = checkFunction<float, K, false>(size / g_cluster_count, g_cluster_count);
                                                result[tid] &= checkFunction<half, K, true>(size / g_cluster_count, g_cluster_count);
                                                }, size, i, thread_result);
                }
                for(int i = 0; i < thread_count; i++) {
                    thread[i]->join();
                    delete thread[i];
                    isPass &= thread_result[i];
                }
            }
            free(thread);
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}

TEST_F(TopkTest, Performance)
{
    std::vector<int64_t> params;
    params.emplace_back(8 * 1024 * 1024);
    params.emplace_back(20 * 1024 * 1024);
    params.emplace_back(40 * 1024 * 1024);

    std::vector<std::string> param_string;
    param_string.push_back("8*1024*1024");
    param_string.push_back("20*1024*1024");
    param_string.push_back("40*1024*1024");

    static const int K = 64;
    std::vector<float> golden_value = {1.2f, 1.6f, 2.2f};
    std::string test_name = "<AETest::TopkTest::Performance";
    try {
        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("-").append(param_string.at(i));
            auto time = checkPerformance<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }

        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("Half").append("-").append(param_string.at(i));
            auto time = checkPerformance<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    } catch(...) {
        std::cout << test_name << " 0.0ms " << "Fail> " << std::endl;
    }
}

TEST_F(TopkTest, Function16)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 16;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}

TEST_F(TopkTest, Function32)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 32;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}

TEST_F(TopkTest, Function128)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 128;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}

TEST_F(TopkTest, Function256)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 256;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}

TEST_F(TopkTest, Function512)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 512;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}
 
TEST_F(TopkTest, Function1024)
{
    std::vector<int> params = {8*1024*1024, 20*1024*1024, 40*1024*1024};
    std::string test_name = "<AETest::TopkTest::Function";
    try {
        const int K = 1024;
        bool isPass = true;
        for(unsigned int i = 0; i < params.size(); ++i) {
            for(int loop = 0; loop < TEST_LOOP_COUNT; ++loop) {
                isPass &= checkFunction<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
                isPass &= checkFunction<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            }
        }

        if(isPass) {
            std::cout << test_name << " 0.0ms " << "Pass>" << std::endl;
        } else {
            std::cout << test_name << " 0.0ms " << "Fail>" << std::endl;
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    }
}


TEST_F(TopkTest, Performance128)
{
    std::vector<int64_t> params;
    params.emplace_back(8 * 1024 * 1024);
    params.emplace_back(20 * 1024 * 1024);
    params.emplace_back(40 * 1024 * 1024);

    std::vector<std::string> param_string;
    param_string.push_back("8*1024*1024");
    param_string.push_back("20*1024*1024");
    param_string.push_back("40*1024*1024");

    static const int K = 128;
    std::vector<float> golden_value = {1.355555f, 1.8f, 2.4f};
    std::string test_name = "<AETest::TopkTest::Performance";
    try {
        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("-").append(param_string.at(i));
            auto time = checkPerformance<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }

        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("Half").append("-").append(param_string.at(i));
            auto time = checkPerformance<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    } catch(...) {
        std::cout << test_name << " 0.0ms " << "Fail> " << std::endl;
    }
}

TEST_F(TopkTest, Performance256)
{
    std::vector<int64_t> params;
    params.emplace_back(8 * 1024 * 1024);
    params.emplace_back(20 * 1024 * 1024);
    params.emplace_back(40 * 1024 * 1024);

    std::vector<std::string> param_string;
    param_string.push_back("8*1024*1024");
    param_string.push_back("20*1024*1024");
    param_string.push_back("40*1024*1024");

    static const int K = 256;
    std::vector<float> golden_value = {1.5f, 2.1f, 2.85f};
    std::string test_name = "<AETest::TopkTest::Performance";
    try {
        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("-").append(param_string.at(i));
            auto time = checkPerformance<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }

        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("Half").append("-").append(param_string.at(i));
            auto time = checkPerformance<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    } catch(...) {
        std::cout << test_name << " 0.0ms " << "Fail> " << std::endl;
    }
}

TEST_F(TopkTest, Performance512)
{
    std::vector<int64_t> params;
    params.emplace_back(8 * 1024 * 1024);
    params.emplace_back(20 * 1024 * 1024);
    params.emplace_back(40 * 1024 * 1024);

    std::vector<std::string> param_string;
    param_string.push_back("8*1024*1024");
    param_string.push_back("20*1024*1024");
    param_string.push_back("40*1024*1024");

    static const int K = 512;
    std::vector<float> golden_value = {2.35f, 3.28f, 4.15f};
    std::string test_name = "<AETest::TopkTest::Performance";
    try {
        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("-").append(param_string.at(i));
            auto time = checkPerformance<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }

        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("Half").append("-").append(param_string.at(i));
            auto time = checkPerformance<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    } catch(...) {
        std::cout << test_name << " 0.0ms " << "Fail> " << std::endl;
    }
}

TEST_F(TopkTest, Performance1024)
{
    std::vector<int64_t> params;
    params.emplace_back(8 * 1024 * 1024);
    params.emplace_back(20 * 1024 * 1024);
    params.emplace_back(40 * 1024 * 1024);

    std::vector<std::string> param_string;
    param_string.push_back("8*1024*1024");
    param_string.push_back("20*1024*1024");
    param_string.push_back("40*1024*1024");

    static const int K = 1024;
    std::vector<float> golden_value = {4.4f, 5.5f, 6.6f};
    std::string test_name = "<AETest::TopkTest::Performance";
    try {
        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("-").append(param_string.at(i));
            auto time = checkPerformance<float, K, false>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }

        for(unsigned int i = 0; i < params.size(); ++i) {
            std::string test_full_name = test_name;
            test_full_name.append("Half").append("-").append(param_string.at(i));
            auto time = checkPerformance<half, K, true>(params.at(i) / g_cluster_count, g_cluster_count);
            std::string time_str = numberToString<float>(time).append("ms");
            if(time < golden_value.at(i)) {
                std::cout << test_full_name << " " << time_str << " Pass>" << std::endl;
            } else {
                std::cout << test_full_name << " " << time_str << " Fail>" << std::endl;
            }
        }
    } catch(std::exception e) {
        std::cout << test_name << " 0.0ms " << "Fail> " << e.what() << std::endl;
    } catch(...) {
        std::cout << test_name << " 0.0ms " << "Fail> " << std::endl;
    }
}
} //AETEST








#include "utils/dl_log.h"
#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

int g_cluster_count = 0;
extern void load_all_cases_fatbin(std::string);
extern void releaseTopkTmpMemory();

int main(int argc, char* argv[])
{
    initDlLogger();
    setDlLoggerSeverity(DLLoggerSeverity_INFO);

    cuInit(0);
    cudaSetDevice(0);
    cudaDeviceProp devide_prop;
    cudaGetDeviceProperties(&devide_prop, 0);
    g_cluster_count = devide_prop.clusterCount;
    load_all_cases_fatbin("./kernels.fatbin");

    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    releaseTopkTmpMemory();
    return ret;
}

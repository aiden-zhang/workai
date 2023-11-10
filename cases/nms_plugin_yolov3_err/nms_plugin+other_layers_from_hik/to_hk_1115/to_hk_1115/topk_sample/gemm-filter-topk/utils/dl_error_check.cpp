






#include "dl_log.h"
#include "dl_error_check.h"


void cuda_check(cudaError_t state, std::string file, int line)
{
    if (state != cudaSuccess) {
        DlLogE << "CUDA Error code num is:" << state;
        DlLogE << "CUDA Error:" << cudaGetErrorString(state);
        DlLogE << "Error location:" << file << ": " << line;
        ASSERT(false);
    }
}

void cudnn_check(cudnnStatus_t state, std::string file, int line)
{
    if (state != CUDNN_STATUS_SUCCESS) {
        DlLogE << "Cudnn Error code num is:" << state;
        //DlLogE << "Cudnn Error:" << cudnnGetErrorString(state);
        DlLogE << "Error location:" << file << ": " << line;
        ASSERT(false);
    }
}

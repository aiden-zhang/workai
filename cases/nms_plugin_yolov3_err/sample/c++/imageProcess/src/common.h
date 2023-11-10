#ifndef COMMON_H_
#define COMMON_H_

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a):(b))
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a):(b))
#define MIN_3(a,b,c) MIN(MIN(a,b),(c))
#endif

#ifndef CLIP
#define CLIP(a,start,end) MAX(MIN((a),(end)),(start))
#endif

#define ALIGN1B 0
#define ALIGN2B 1
#define ALIGN4B 2
#define ALIGN8B 3


// ((uint64_t)global & 7) == 0 and (size & 7) == 0
__device__ __forceinline__ void global2share_copy_align(const uint8_t *global, uint8_t *sm, int size) {
    int idx = threadIdx.x;
    int copy_num = size >> 3;
    while (idx < copy_num) {
        ((uint64_t*)sm)[idx] = ((uint64_t*)global)[idx];
        idx += blockDim.x;
    }
}

/*
Usage:
int offset = global2share_copy(global, sm, size);   
sm += sm;
*/
__device__ __forceinline__ uint8_t global2share_copy(const uint8_t *global, uint8_t *sm, int size) {
    int idx = threadIdx.x;
    uint8_t front = (8 - ((uint64_t)global & 7)) & 7;
    uint8_t back = (size - front) & 7;
    int copy_num = (size - front - back) >> 3;
    while (idx < copy_num) {
        ((uint64_t*)sm)[idx + 1] = ((uint64_t*)(global + front))[idx];
        idx += blockDim.x;
    }
    if (threadIdx.x < front) {
        sm[8 - front + threadIdx.x] = global[threadIdx.x];
    }
    if (threadIdx.x < back) {
        sm[8 + (copy_num << 3) + threadIdx.x] = global[size - back + threadIdx.x];
    }
    return 8 - front;
}

#define CUDA_CHECK(state)                                                       \
    do {                                                                        \
      if (state != cudaSuccess) {                                               \
        std::cout << "CUDA Error code num is:" << state << std::endl;           \
        std::cout << "CUDA Error:" << cudaGetErrorString(state) << std::endl;   \
        std::cout << __FILE__ << " " << __LINE__ << "line!" << std::endl;       \
        abort();                                                                \
      }                                                                         \
    } while (0)
#endif /* COMMON_H_ */

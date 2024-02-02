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

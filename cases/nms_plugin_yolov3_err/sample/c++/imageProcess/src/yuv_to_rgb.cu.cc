#include "cuda_fp16.h"
#include "yuv_to_rgb.h"

void YUVYu12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  dim3 block(32, 4, 1);
  dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
  Yuv2rgb24<true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
}

void YUVNv12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  dim3 block(32, 4, 1);
  dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
  Yuv2rgb24<false><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
}

void YUVYu12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  dim3 block(32, 4, 1);
  dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
  Yuv2rgb24_plane<true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (uint32_t*)out_buf, in_h >> 1, in_w >> 2);
}

void YUVNv12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  dim3 block(32, 4, 1);
  dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
  Yuv2rgb24_plane<false><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (uint32_t*)out_buf, in_h >> 1, in_w >> 2);
}

void YUVYu12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  dim3 block(32, 4, 1);
  dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
  Yuv2rgb24<true, true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
}

void YUVNv12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  dim3 block(32, 4, 1);
  dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
  Yuv2rgb24<false, true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
}

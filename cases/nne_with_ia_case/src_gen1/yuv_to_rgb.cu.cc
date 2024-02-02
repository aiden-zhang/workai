#include "cuda_fp16.h"
#include "yuv_to_rgb.h"
#include <stdio.h>

void YUVYu12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                     int in_w, int in_h, cudaStream_t stream){
  if (in_w % 4 == 0) {
      dim3 block(32, 4, 1);
      dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
      Yuv2rgb24<true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
  } else {
      dim3 block(256, 1, 1);
      dim3 grid(in_h / 2, 1, 1);
      int sm_size = in_w * 6 + 16 * 3;
      Yuv2rgb24_general<true, false, uint8_t, false><<<grid, block, sm_size, stream>>>(in_buf, out_buf, in_h, in_w);
  }
}

void YUVNv12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  if (in_w % 4 == 0) {
      dim3 block(32, 4, 1);
      dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
      Yuv2rgb24<false><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
  } else {
      dim3 block(256, 1, 1);
      dim3 grid(in_h / 2, 1, 1);
      int sm_size = in_w * 6 + 16 * 3;
      Yuv2rgb24_general<false, false, uint8_t, false><<<grid, block, sm_size, stream>>>(in_buf, out_buf, in_h, in_w);
  }
}

void YUVYu12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  if (in_w % 4 == 0) {
      dim3 block(32, 4, 1);
      dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
      Yuv2rgb24_plane<true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (uint32_t*)out_buf, in_h >> 1, in_w >> 2);
  } else {
      dim3 block(256, 1, 1);
      dim3 grid(in_h / 2, 1, 1);
      int sm_size = in_w * 6 + 16 * 3;
      Yuv2rgb24_general<true, false, uint8_t, true><<<grid, block, sm_size, stream>>>(in_buf, out_buf, in_h, in_w);
  }
}

void YUVNv12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  if (in_w % 4 == 0) {
      dim3 block(32, 4, 1);
      dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
      Yuv2rgb24_plane<false><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (uint32_t*)out_buf, in_h >> 1, in_w >> 2);
  } else {
      dim3 block(256, 1, 1);
      dim3 grid(in_h / 2, 1, 1);
      int sm_size = in_w * 6 + 16 * 3;
      Yuv2rgb24_general<false, false, uint8_t, true><<<grid, block, sm_size, stream>>>(in_buf, out_buf, in_h, in_w);
  }
}

void YUVYu12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  if (in_w % 4 == 0) {
      dim3 block(32, 4, 1);
      dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
      Yuv2rgb24<true, true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
  } else {
      dim3 block(256, 1, 1);
      dim3 grid(in_h / 2, 1, 1);
      int sm_size = in_w * 6 + 16 * 3;
      Yuv2rgb24_general<true, true, float, false><<<grid, block, sm_size, stream>>>(in_buf, out_buf, in_h, in_w);
  }
}

void YUVNv12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
  if (in_w % 4 == 0) {
      dim3 block(32, 4, 1);
      dim3 grid(((in_w >> 2) + block.x - 1) / block.x, ((in_h >> 1) + block.y - 1) / block.y, 1);
      Yuv2rgb24<false, true><<<grid, block, 0, stream>>>((uint32_t*)in_buf, (void*)out_buf, in_h >> 1, in_w >> 2);
  } else {
      dim3 block(256, 1, 1);
      dim3 grid(in_h / 2, 1, 1);
      int sm_size = in_w * 6 + 16 * 3;
      Yuv2rgb24_general<false, true, float, false><<<grid, block, sm_size, stream>>>(in_buf, out_buf, in_h, in_w);
  }
}

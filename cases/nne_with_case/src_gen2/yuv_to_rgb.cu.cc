#include "cuda_fp16.h"
#include "yuv_to_rgb.h"
#include <stdio.h>

void YUVYu12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                     int in_w, int in_h, cudaStream_t stream){
    dim3 block(32, 4, 1);
    dim3 grid((in_w / 2 + block.x - 1) / block.x, (in_h + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    YU122rgb24_general<uint8_t, false, 2><<<grid, block, 0, stream>>>(in_Y, in_U, in_V, out_buf, in_h, in_w);
}

void YUVNv12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
    dim3 block(32, 4, 1);
    dim3 grid((in_w / 2 + block.x - 1) / block.x, (in_h + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_Y + in_h * in_w;
    NV122rgb24_general<uint8_t, false, 2><<<grid, block, 0, stream>>>(in_Y, in_UV, out_buf, in_h, in_w);
}

void YUVYu12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
    dim3 block(32, 4, 1);
    dim3 grid((in_w / 4 + block.x - 1) / block.x, (in_h + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    YU122rgb24_general<uint8_t, true, 4><<<grid, block, 0, stream>>>(in_Y, in_U, in_V, out_buf, in_h, in_w);
}

void YUVNv12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
    dim3 block(32, 4, 1);
    dim3 grid((in_w / 4 + block.x - 1) / block.x, (in_h + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_Y + in_h * in_w;
    NV122rgb24_general<uint8_t, true, 4><<<grid, block, 0, stream>>>(in_Y, in_UV, out_buf, in_h, in_w);
}

void YUVYu12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
    dim3 block(32, 1, 1);
    dim3 grid((in_w + block.x - 1) / block.x, in_h ,1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    YU122rgb24_general<float, false, 1><<<grid, block, 0, stream>>>(in_Y, in_U, in_V, out_buf, in_h, in_w);
}

void YUVNv12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream){
    dim3 block(32, 1, 1);
    dim3 grid((in_w + block.x - 1) / block.x, in_h, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_Y + in_h * in_w;
    NV122rgb24_general<float, false, 1><<<grid, block, 0, stream>>>(in_Y, in_UV, out_buf, in_h, in_w);
}

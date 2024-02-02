#include <stdio.h>
#include "cuda_fp16.h"
#include "yuv_resize.h"

void YUVNv12ResizeNearest(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768,1,1);
    if(out_w < 768) block.x = out_w;
    dim3 grid(out_h / 2, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (in_w & 7) == 0) {
        resize_kernel_nv12_nv21<true><<<grid, block, (in_w + 16) * 3, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    } else {
        resize_kernel_nv12_nv21<false><<<grid, block, (in_w + 16) * 3, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    }
}

void YUVNv21ResizeNearest(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768,1,1);
    if(out_w < 768) block.x = out_w;
    dim3 grid(out_h / 2, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (in_w & 7) == 0) {
        resize_kernel_nv12_nv21<true><<<grid, block, (in_w + 16) * 3, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    } else {
        resize_kernel_nv12_nv21<false><<<grid, block, (in_w + 16) * 3, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    }
}

void YUVI420ResizeNearest(uint8_t* in_buf,uint8_t* out_buf,int in_w,int in_h,int out_w,int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768,1,1);
    if(out_w < 768) block.x = out_w;
    dim3 grid(out_h / 2, 1, 1);
    if (((uint64_t)in_buf & 15) == 0 && (in_w & 15) == 0) {
        resize_kernel_i420<true><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    } else {
        resize_kernel_i420<false><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    }
}

void YUVNv12ResizeBilinear(uint8_t* in_buf,uint8_t* out_buf,int in_w,int in_h,int out_w,int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768,1,1);
    if(out_w < 768) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (in_w & 7) == 0) {
        bilinear_resize_kernel_nv12_nv21<true><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    } else {
        bilinear_resize_kernel_nv12_nv21<false><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    }
}

void YUVNv21ResizeBilinear(uint8_t* in_buf,uint8_t* out_buf,int in_w,int in_h,int out_w,int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768,1,1);
    if(out_w < 768) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (in_w & 7) == 0) {
        bilinear_resize_kernel_nv12_nv21<true><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    } else {
        bilinear_resize_kernel_nv12_nv21<false><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    }
}

void YUVI420ResizeBilinear(uint8_t* in_buf,uint8_t* out_buf,int in_w,int in_h,int out_w,int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768,1,1);
    if(out_w < 768) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    if (((uint64_t)in_buf & 15) == 0 && (in_w & 15) == 0) {
        bilinear_resize_kernel_i420<true><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    } else {
        bilinear_resize_kernel_i420<false><<<grid, block, (in_w + 16) * 4, stream>>>(in_buf, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans, out_buf);
    }
}

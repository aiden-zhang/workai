#include <stdio.h>
#include "cuda_fp16.h"
#include "yuv_resize.h"

void YUVNv12ResizeNearest(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768, 1, 1);
    if (out_w < 768)
        block.x = out_w;
    dim3 grid((out_h / 2 * out_w + block.x - 1) / block.x, 1, 1);

    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_buf + in_h * in_w;
    uint8_t *out_Y = out_buf;
    uint8_t *out_UV = out_buf + out_h * out_w;
    nearest_resize_kernel_nv12_nv21<<<grid, block, 0, stream>>>(in_Y, in_UV, out_Y, out_UV, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans);
}

void YUVNv21ResizeNearest(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768, 1, 1);
    if (out_w < 768)
        block.x = out_w;
    dim3 grid((out_h / 2 * out_w + block.x - 1) / block.x, 1, 1);

    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_buf + in_h * in_w;
    uint8_t *out_Y = out_buf;
    uint8_t *out_UV = out_buf + out_h * out_w;
    nearest_resize_kernel_nv12_nv21<<<grid, block, 0, stream>>>(in_Y, in_UV, out_Y, out_UV, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans);
}

void YUVI420ResizeNearest(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(768, 1, 1);
    if (out_w < 768)
        block.x = out_w;
    dim3 grid((out_h / 2 * out_w + block.x - 1) / block.x, 1, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    uint8_t *out_Y = out_buf;
    uint8_t *out_U = out_Y + out_h * out_w;
    uint8_t *out_V = out_U + (out_h * out_w >> 2);
    nearest_resize_kernel_i420<<<grid, block, 0, stream>>>(in_Y, in_U, in_V, out_Y, out_U, out_V, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans);
}

void YUVNv12ResizeBilinear(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(32, 16, 1);
    dim3 grid((out_w / 2 + block.x - 1) / block.x, (out_h / 2 + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_buf + in_h * in_w;
    uint8_t *out_Y = out_buf;
    uint8_t *out_UV = out_buf + out_h * out_w;
    bilinear_resize_kernel_nv12_nv21<<<grid, block, 0, stream>>>(in_Y, in_UV, out_Y, out_UV, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans);
}

void YUVNv21ResizeBilinear(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(32, 16, 1);
    dim3 grid((out_w / 2 + block.x - 1) / block.x, (out_h / 2 + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_UV = in_buf + in_h * in_w;
    uint8_t *out_Y = out_buf;
    uint8_t *out_UV = out_buf + out_h * out_w;
    bilinear_resize_kernel_nv12_nv21<<<grid, block, 0, stream>>>(in_Y, in_UV, out_Y, out_UV, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans);
}

void YUVI420ResizeBilinear(uint8_t* in_buf, uint8_t* out_buf, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
{
    float t_h_trans = in_h * 1.0f / out_h;
    float t_w_trans = in_w * 1.0f / out_w;

    dim3 block(32, 16, 1);
    dim3 grid((out_w / 2 + block.x - 1) / block.x, (out_h / 2 + block.y - 1) / block.y, 1);
    uint8_t *in_Y = in_buf;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    uint8_t *out_Y = out_buf;
    uint8_t *out_U = out_Y + out_h * out_w;
    uint8_t *out_V = out_U + (out_h * out_w >> 2);
    bilinear_resize_kernel_i420<<<grid, block, 0, stream>>>(in_Y, in_U, in_V, out_Y, out_U, out_V, in_h, in_w, out_h, out_w, t_h_trans, t_w_trans);
}

#include "yuv2rgb_affine.h"
#include <stdio.h>

void YU122RGBAffine(uint8_t* input, uint8_t* output,
        int in_w, int in_h, int out_w, int out_h, float m[6], cudaStream_t stream) {
    dim3 block(128, 1, 1);
    dim3 grid((out_h * out_w + block.x - 1) / block.x, 1, 1);
    uint8_t *in_Y = input;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);

    float alpha = 0.f;
    if (m[0] != 0.f) {
        alpha = fabs(m[3] / m[0]);
        alpha = atanf(alpha);
        alpha = alpha / 3.14f * 180;
    }
    int stride = out_h;
    if (alpha > 40)
    {
        stride = 1;
    }
    yuv2rgb_affine_kernel<false><<<grid, block, 0, stream>>>(
        in_Y, in_U, in_V, output, in_w, in_h, out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5], stride);
}

void RoiNv122RGBAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(128, 1, 1);
    if (block.x >  out_w) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    uint8_t *in_Y = input;
    uint8_t *in_UV = in_Y + in_h * in_w;

    roi_nv122rgb_affine_norm_kernel<false><<<grid, block, 0, stream>>>(
        in_Y, in_UV, output, in_w, in_h, roi_w_start, roi_h_start, roi_w,
        roi_h, out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5],
        mean1, mean2, mean3, std1, std2, std3, scale, pad1, pad2, pad3);
}

void RoiYU122RGBAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(128, 1, 1);
    if (block.x >  out_w) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    uint8_t *in_Y = input;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    roi_yu122rgb_affine_norm_kernel<false, true><<<grid, block, 0, stream>>>(
        in_Y, in_U, in_V, output, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h,
        out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5],
        mean1, mean2, mean3, std1, std2, std3, scale, pad1, pad2, pad3);
}

void RoiYU122BGRAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(128, 1, 1);
    if (block.x >  out_w) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    uint8_t *in_Y = input;
    uint8_t *in_U = in_Y + in_h * in_w;
    uint8_t *in_V = in_U + (in_h * in_w >> 2);
    roi_yu122rgb_affine_norm_kernel<false, false><<<grid, block, 0, stream>>>(
        in_Y, in_U, in_V, output, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h,
        out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5],
        mean1, mean2, mean3, std1, std2, std3, scale, pad1, pad2, pad3);
}

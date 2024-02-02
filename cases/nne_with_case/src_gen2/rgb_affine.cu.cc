#include "image_proc.h"
#include "rgb_affine.h"
#include <stdio.h>

void GrayAffine(uint8_t *input, uint8_t *output, int in_w, int in_h,
            int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h, 
            float m[6], cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    rgb24_affine_kernel<false><<<grid, block, 0, stream>>>(input, output,
              in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h,
              m[0], m[1], m[2], m[3], m[4], m[5]);   
}

void RGBAffine(uint8_t* input, uint8_t* output, int in_w, int in_h, 
    int roi_w_start, int roi_h_start, int roi_w, int roi_h,
    int out_w, int out_h, float m[6], cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    rgb24_affine_kernel<true><<<grid, block, 0, stream>>>(input, output,
              in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h,
              m[0], m[1], m[2], m[3], m[4], m[5]);   
}

void BatchRGBAffine(uint8_t* input, uint8_t* output, int batch, int out_batch_offset, int in_w, int in_h,
        RGBAffineInfo_t *param, cudaStream_t stream) {
  dim3 block(128, 1, 1);
  dim3 grid(batch, 32, 1);
  if (batch < 4) grid.y = 128 / batch;
  batch_rgb24_affine_kernel<<<grid, block, 0, stream>>>(input, output, out_batch_offset,
              in_w, in_h, param);
}

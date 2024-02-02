#include <stdio.h>
#include <assert.h>
#include "cuda_fp16.h"
#include "gray_resize.h"

void GrayResizeBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid((w_out + block.x - 1) / block.x, h_out, 1);
    gray_resize_kernel<true><<<grid, block, 0, stream>>>(in_buf, h_in, w_in, h_out, w_out, 1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
}

void GrayResizeNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if( w_out < 768) block.x = w_out;
    dim3 grid((w_out + block.x - 1) / block.x, h_out, 1);
    gray_resize_kernel<false><<<grid, block, 0, stream>>>(in_buf, h_in, w_in, h_out, w_out, 1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
}

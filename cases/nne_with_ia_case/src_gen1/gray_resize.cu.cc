#include <stdio.h>
#include <assert.h>
#include "cuda_fp16.h"
#include "gray_resize.h"

void GrayResizeBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        gray_resize_kernel<true, true><<<grid,block,w_in * 2 + 16,stream>>>(in_buf,h_in,w_in,h_out,w_out, 1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        gray_resize_kernel<true, false><<<grid,block,w_in * 2 + 16,stream>>>(in_buf,h_in,w_in,h_out,w_out, 1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void GrayResizeNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if( w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        gray_resize_kernel<false, true><<<grid,block,w_in + 16, stream>>>(in_buf,h_in,w_in,h_out,w_out, 1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }else {
        gray_resize_kernel<false, false><<<grid,block,w_in + 16, stream>>>(in_buf,h_in,w_in,h_out,w_out, 1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

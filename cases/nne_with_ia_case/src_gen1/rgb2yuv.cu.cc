#include <stdio.h>
#include "rgb2yuv.h"

void RGB2YU12(uint8_t *in_buf, uint8_t *out_buf, int w, int h, cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(h/2, 1, 1);
    int sm_size = w * 6 + 16;
    if (((uint64_t)in_buf & 7) == 0 && (w & 7) == 0) {
        rgb2yu12_kernel<true><<<grid, block, sm_size, stream>>>(in_buf, out_buf, w, h);
    } else {
        rgb2yu12_kernel<false><<<grid, block, sm_size, stream>>>(in_buf, out_buf, w, h);
    }
}

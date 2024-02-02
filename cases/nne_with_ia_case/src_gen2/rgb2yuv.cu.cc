#include <stdio.h>
#include "rgb2yuv.h"

void RGB2YU12(uint8_t *in_buf, uint8_t *out_buf, int w, int h, cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((w / 2 + block.x - 1) / block.x, h / 2, 1);
    uint8_t *out_u = out_buf + h * w;
    uint8_t *out_v = out_u + (h * w >> 2);
    rgb2yu12_kernel<<<grid, block, 0, stream>>>((uchar3 *)in_buf, out_buf, out_u, out_v, w, h);
}

#include <stdio.h>
#include "rgb_crop.h"

void RGBCrop(uint8_t *in_buf, uint8_t *out_buf,
            int start_w, int start_h, int w_in, 
            int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(128, 1, 1);
    dim3 grid((w_out + block.x - 1) / block.x, h_out, 1);
    rgb_crop_kernel<<<grid, block, 0, stream>>>((uchar3*)in_buf, (uchar3*)out_buf, start_h, start_w, h_in, w_in, h_out, w_out);

}

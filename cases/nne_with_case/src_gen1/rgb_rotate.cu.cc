#include <stdio.h>
#include "rgb_rotate.h"
#include "image_proc.h"
void RGBRotate(uint8_t *in, uint8_t *out, int in_w, int in_h, int rotate_mode, cudaStream_t stream) {
    if (rotate_mode == ROTATE_90_CLOCKWISE) {
        dim3 blocks(TILE_HW, TILE_HW, 1);
        dim3 grids((in_h + TILE_HW - 1) / TILE_HW, (in_w + TILE_HW - 1) / TILE_HW, 1);
        rgb_rotate_90<uchar3><<<grids, blocks, 0, stream>>>((uchar3*)in, (uchar3*)out, in_h, in_w);
    } else if (rotate_mode == ROTATE_180_CLOCKWISE) {
        int in_c = 3;
        dim3 blocks(TILE_HW * in_c, TILE_HW, 1);
        dim3 grids((in_w + TILE_HW - 1) / TILE_HW, (in_h + TILE_HW - 1) / TILE_HW, 1);
        rgb_rotate_180<uint8_t><<<grids, blocks, 0, stream>>>(in, out, in_h, in_w, in_w * in_c);
    } else if (rotate_mode == ROTATE_270_CLOCKWISE){
        dim3 blocks(TILE_HW, TILE_HW, 1);
        dim3 grids((in_h + TILE_HW - 1) / TILE_HW, (in_w + TILE_HW - 1) / TILE_HW, 1);
        rgb_rotate_270<uchar3><<<grids, blocks, 0, stream>>>((uchar3*)in, (uchar3*)out, in_h, in_w);
    } else {
        printf("error:%s %d not support the value of angle\n", __FILE__, __LINE__);
        abort();
    }
}

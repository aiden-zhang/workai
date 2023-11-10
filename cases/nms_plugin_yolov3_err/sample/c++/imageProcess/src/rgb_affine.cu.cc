#include "rgb_affine.h"
#include <stdio.h>

void GrayAffine(uint8_t *input, uint8_t *output, int in_w, int in_h,
            int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h, 
            float m[6], cudaStream_t stream) {
    dim3 block(768, 1, 1);
    if (block.x > out_w) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    gray_affine_kernel<<<grid, block, 0, stream>>>(input, output,
            in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h,
            m[0], m[1], m[2], m[3], m[4], m[5]);

}

void RGBAffine(uint8_t* input, uint8_t* output, int in_w, int in_h, 
    int roi_w_start, int roi_h_start, int roi_w, int roi_h,
    int out_w, int out_h, float m[6], cudaStream_t stream) {
  dim3 block(256,1,1);
  if (block.x >  out_w) block.x = out_w;
  dim3 grid(out_h, 1, 1);
  int sm_size = out_w * 13 * sizeof(int) + out_w * sizeof(uint8_t) + sizeof(int);
  if (sm_size > 48 * 1024) {
    printf("\nERROR:%s:%d\n", __FILE__, __LINE__);
    abort();
  }
  rgb24_affine_kernel<<<grid, block, sm_size, stream>>>(input, output,
            in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h,
            m[0], m[1], m[2], m[3], m[4], m[5]);
}


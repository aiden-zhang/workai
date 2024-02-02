#include "yuv2rgb_affine.h"
#include <stdio.h>

void YU122RGBAffine(uint8_t* input, uint8_t* output,
        int in_w, int in_h, int out_w, int out_h, float m[6], cudaStream_t stream) {
    dim3 block(256,1,1);
    if (block.x >  out_w) block.x = out_w;
    dim3 grid(out_h, 1, 1);
    int sm_size = out_w * 13 * sizeof(int) + out_w * 12 * sizeof(uint8_t) + sizeof(int);
    if (sm_size > 48 * 1024) {
        printf("\nERROR:%s:%d\n", __FILE__, __LINE__);
        abort();
    }
    yuv2rgb_affine_kernel<false><<<grid, block, sm_size, stream>>>(input, output, in_w, 
        in_h, out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5]); 
}

void RoiNv122RGBAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (block.x >  out_w) block.x = out_w;
    int rows = 512 / out_w;
    if (rows > 1) {
        while (rows > 1) {
            if (out_h % rows == 0) {
                break;
            } else {
                rows--;
            }
        }
        block.y = rows;
    }
    dim3 grid(1, out_h / block.y, 1);
    int sm_size = (out_w * 13 * sizeof(int) + out_w * 12 * sizeof(uint8_t) + sizeof(int)) * block.y;
    if (sm_size > 48 * 1024) {
        printf("\nERROR:%s:%d\n", __FILE__, __LINE__);
        abort();
    }
    roi_nv122rgb_affine_norm_kernel<false><<<grid, block, sm_size, stream>>>(input, output, in_w, 
        in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5],
        mean1, mean2, mean3, std1, std2, std3, scale, pad1, pad2, pad3); 
}

void RoiYU122RGBAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (block.x >  out_w) block.x = out_w;
    int rows = 512 / out_w;
    if (rows > 1) {
        while (rows > 1) {
            if (out_h % rows == 0) {
                break;
            } else {
                rows--;
            }
        }
        block.y = rows;
    }
    dim3 grid(1, out_h / block.y, 1);
    int sm_size = (out_w * 13 * sizeof(int) + out_w * 12 * sizeof(uint8_t) + sizeof(int)) * block.y;
    if (sm_size > 48 * 1024) {
        printf("\nERROR:%s:%d\n", __FILE__, __LINE__);
        abort();
    }
    roi_yu122rgb_affine_norm_kernel<false, true><<<grid, block, sm_size, stream>>>(input, output, in_w, 
        in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5],
        mean1, mean2, mean3, std1, std2, std3, scale, pad1, pad2, pad3); 
}

void RoiYU122BGRAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (block.x >  out_w) block.x = out_w;
    int rows = 512 / out_w;
    if (rows > 1) {
        while (rows > 1) {
            if (out_h % rows == 0) {
                break;
            } else {
                rows--;
            }
        }
        block.y = rows;
    }
    dim3 grid(1, out_h / block.y, 1);
    int sm_size = (out_w * 13 * sizeof(int) + out_w * 12 * sizeof(uint8_t) + sizeof(int)) * block.y;
    if (sm_size > 48 * 1024) {
        printf("\nERROR:%s:%d\n", __FILE__, __LINE__);
        abort();
    }
    roi_yu122rgb_affine_norm_kernel<false, false><<<grid, block, sm_size, stream>>>(input, output, in_w, 
        in_h, roi_w_start, roi_h_start, roi_w, roi_h, out_w, out_h, m[0], m[1], m[2], m[3], m[4], m[5],
        mean1, mean2, mean3, std1, std2, std3, scale, pad1, pad2, pad3); 
}

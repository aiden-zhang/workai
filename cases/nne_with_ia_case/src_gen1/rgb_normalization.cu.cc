#include "cuda_fp16.h"
#include "rgb_normalization.h"

void RGBNormalization(uint8_t* in_buf, float* out_buf,
    int in_w, int in_h, int in_c, float mean, float standard, float scale, cudaStream_t stream){
    int size = in_w * in_h * in_c;
    dim3 block(256, 1, 1);
    dim3 grid((size + block.x - 1) / block.x, 1, 1);
    rgb_normalization_kernel<uint8_t, float><<<grid, block, 0, stream>>>(in_buf, out_buf, mean, standard, scale, size);
}

void RGBNormalization_3Channels(uint8_t* in_buf, float* out_buf, 
    int in_w, int in_h, float mean1, float mean2, float mean3,
    float standard1, float standard2, float standard3, float scale, bool input_plane, bool output_plane, bool channel_rev, cudaStream_t stream){
    dim3 block(512,1,1);
    if(in_w < 512) block.x = in_w;
    dim3 grid(in_h, 1, 1);
    bool input_align = ((uint64_t)in_buf & 7) == 0 && (in_w & 7) == 0 ? true : false;
    int sm_size = (in_w + 16) * 3;
    if (channel_rev) {
        if (input_plane) {
            if (output_plane) {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, true, true, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, true, false, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            } else {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, false, true, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, false, false, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            }
        } else {
            if (output_plane) {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, true, true, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, true, false, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            } else {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, false, true, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, false, false, 2><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            }
        }
    } else {
        if (input_plane) {
            if (output_plane) {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, true, true, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, true, false, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            } else {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, false, true, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, true, false, false, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            }
        } else {
            if (output_plane) {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, true, true, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, true, false, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            } else {
                if (input_align) {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, false, true, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                } else {
                    rgb_normalization_3channels_kernel<uint8_t, float, false, false, false, 0><<<grid, block, sm_size, stream>>>(
                            in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h);
                }
            }
        }
    }
}

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
    dim3 block(128, 1, 1);
    dim3 grid((in_w + block.x - 1) / block.x, in_h, 1);
    int rev = channel_rev ? 2 : 0;
    if (input_plane) {
        if (output_plane) {
            rgb_normalization_3channels_kernel<uint8_t, float, true, true><<<grid, block, 0, stream>>>(
                    in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h, rev);
        } else {
            rgb_normalization_3channels_kernel<uint8_t, float, true, false><<<grid, block, 0, stream>>>(
                    in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h, rev);
        }
    } else {
        if (output_plane) {
            rgb_normalization_3channels_kernel<uint8_t, float, false, true><<<grid, block, 0, stream>>>(
                    in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h, rev);
        } else {
            rgb_normalization_3channels_kernel<uint8_t, float, false, false><<<grid, block, 0, stream>>>(
                    in_buf, out_buf, mean1, mean2, mean3, standard1, standard2, standard3, scale, in_w, in_h, rev);
        }
    }
}

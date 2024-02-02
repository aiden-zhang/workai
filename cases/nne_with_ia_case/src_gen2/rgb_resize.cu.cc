#include <stdio.h>
#include "cuda_fp16.h"
#include "rgb_resize.h"

void RGBResizeBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768, 1, 1);
    if (w_out < 768) block.x = w_out;
    dim3 grid((h_out * w_out + block.x - 1) / block.x, 1, 1);
    rgb_resize_kernel<true><<<grid, block, 0, stream>>>((uchar3 *)in_buf, h_in, w_in, h_out, w_out,
                                                        1.0f * h_in / h_out, 1.0f * w_in / w_out, (uchar3 *)out_buf);
}

void RGBResizeNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768, 1, 1);
    if (w_out < 768) block.x = w_out;
    dim3 grid((h_out * w_out + block.x - 1) / block.x, 1, 1);
    rgb_resize_kernel<false><<<grid, block, 0, stream>>>((uchar3 *)in_buf, h_in, w_in, h_out, w_out,
                                                        1.0f * h_in / h_out, 1.0f * w_in / w_out, (uchar3 *)out_buf);
}

void RGBResizePlanePadNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, int w_box, int h_box, int w_b, int h_b, cudaStream_t stream) {
    dim3 block(32, 16, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
    rgb_plane_resize_pad_kernel<false><<<grid, block, 0, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_box, w_box, h_b, w_b,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
}

void RGBResizePlanePadBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, int w_box, int h_box, int w_b, int h_b, cudaStream_t stream) {
    dim3 block(32, 16, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
    rgb_plane_resize_pad_kernel<true><<<grid, block, 0, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_box, w_box, h_b, w_b,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
}

void RGBResizePlaneNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(32, 16, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
    rgb_plane_resize_pad_kernel<false><<<grid, block, 0, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_out, w_out, 0, 0,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
}

void RGBResizePlaneBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(32, 16, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
    rgb_plane_resize_pad_kernel<true><<<grid, block, 0, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_out, w_out, 0, 0,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
}

void RGBResizeWithROIBilinear(uint8_t *in_buf, uint8_t *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h, cudaStream_t stream) {
    float w_scale = 1.0f * roi_w / w_out;
    float h_scale = 1.0f * roi_h / h_out;
    dim3 block(32, 16, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
    rgb_resize_ROI_kernel<true><<<grid, block, 0, stream>>>((uchar3*)in_buf, h_in, w_in, h_out, w_out, h_scale, w_scale, 
            roi_h_start, roi_w_start, roi_h, roi_w, (uchar3*)out_buf);
}

void RGBResizeWithROINearest(uint8_t *in_buf, uint8_t *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h, cudaStream_t stream) {

    float w_scale = 1.0f * roi_w / w_out;
    float h_scale = 1.0f * roi_h / h_out;
    dim3 block(32, 16, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
    rgb_resize_ROI_kernel<false><<<grid, block, 0, stream>>>((uchar3*)in_buf, h_in, w_in, h_out, w_out, h_scale, w_scale, 
            roi_h_start, roi_w_start, roi_h, roi_w, (uchar3*)out_buf);
}



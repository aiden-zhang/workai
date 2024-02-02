#include <stdio.h>
#include "cuda_fp16.h"
#include "rgb_resize.h"

void RGBResizeBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        rgb_resize_kernel<true, 3, true><<<grid, block, w_in * 3 * 2 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        rgb_resize_kernel<true, 3, false><<<grid, block, w_in * 3 * 2 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void RGBResizeNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        rgb_resize_kernel<false, 3, true><<<grid, block, w_in * 3 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        rgb_resize_kernel<false, 3, false><<<grid, block, w_in * 3 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void RGBResizePlanePadNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, int w_box, int h_box, int w_b, int h_b, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        rgb_plane_resize_pad_kernel<false, true><<<grid, block, w_in + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_box, w_box, h_b, w_b,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        rgb_plane_resize_pad_kernel<false, false><<<grid, block, w_in + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_box, w_box, h_b, w_b,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void RGBResizePlanePadBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, int w_box, int h_box, int w_b, int h_b, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        rgb_plane_resize_pad_kernel<true, true><<<grid, block, w_in * 2 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_box, w_box, h_b, w_b,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        rgb_plane_resize_pad_kernel<true, false><<<grid, block, w_in * 2 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_box, w_box, h_b, w_b,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void RGBResizePlaneNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        rgb_plane_resize_pad_kernel<false, true><<<grid, block, w_in + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_out, w_out, 0, 0,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        rgb_plane_resize_pad_kernel<false, false><<<grid, block, w_in + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_out, w_out, 0, 0,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void RGBResizePlaneBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (w_in & 7) == 0) {
        rgb_plane_resize_pad_kernel<true, true><<<grid, block, w_in * 2 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_out, w_out, 0, 0,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    } else {
        rgb_plane_resize_pad_kernel<true, false><<<grid, block, w_in * 2 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_out, w_out, 0, 0,
            1.0f * h_in / h_out, 1.0f * w_in / w_out, out_buf);
    }
}

void RGBResizeWithROIBilinear(uint8_t *in_buf, uint8_t *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h, cudaStream_t stream) {
    float w_scale = 1.0f * roi_w / w_out;
    float h_scale = 1.0f * roi_h / h_out; 
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0 && (w_in & 7) == 0) {
        rgb_resize_ROI_kernel<true, 3, true><<<grid, block, roi_w * 3 * 2 + 16 * 2, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_scale, w_scale, 
                roi_h_start, roi_w_start, roi_h, roi_w, out_buf);
    } else {
        rgb_resize_ROI_kernel<true, 3, false><<<grid, block, roi_w * 3 * 2 + 16 * 2, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_scale, w_scale, 
                roi_h_start, roi_w_start, roi_h, roi_w, out_buf);
    }
}

void RGBResizeWithROINearest(uint8_t *in_buf, uint8_t *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h, cudaStream_t stream) {

    float w_scale = 1.0f * roi_w / w_out;
    float h_scale = 1.0f * roi_h / h_out; 
    dim3 block(768,1,1);
    if(w_out < 768) block.x = w_out;
    dim3 grid(h_out, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0 && (w_in & 7) == 0) {
        rgb_resize_ROI_kernel<false, 3, true><<<grid, block, roi_w * 3 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_scale, w_scale, 
                roi_h_start, roi_w_start, roi_h, roi_w, out_buf);
    } else {
        rgb_resize_ROI_kernel<false, 3, false><<<grid, block, roi_w * 3 + 16, stream>>>(in_buf, h_in, w_in, h_out, w_out, h_scale, w_scale, 
                roi_h_start, roi_w_start, roi_h, roi_w, out_buf);
    }
}



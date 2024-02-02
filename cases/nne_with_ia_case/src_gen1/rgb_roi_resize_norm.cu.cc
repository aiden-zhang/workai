#include <stdio.h>
#include "cuda_fp16.h"
#include "image_proc.h"
#include "rgb_roi_resize_norm.h"

#define LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(is_bilinear, align_in_w, channel_rev) \
    dim3 block(512,1,1); \
    if(w_out < 512) block.x = w_out; \
    dim3 grid(h_out, 1, 1); \
    switch (h_out % 4) { \
        case 0: \
            grid.x = is_bilinear ? h_out / (1 << ALIGN4B) : grid.x; \
            rgb_resize_ROI_norm_kernel<is_bilinear, align_in_w, 1 << ALIGN4B, channel_rev><<<grid, block, roi_w * 3 * 2 + 16 * 2, stream>>>( \
                       in_buf, out_buf, h_in, w_in, h_out, w_out, img_h, img_w, pad_h, pad_w, h_scale, w_scale, \
                       roi_h_start, roi_w_start, roi_h, roi_w, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3); \
            break; \
        case 2: \
            grid.x = is_bilinear ? h_out / (1 << ALIGN2B) : grid.x; \
            rgb_resize_ROI_norm_kernel<is_bilinear, align_in_w, 1 << ALIGN2B, channel_rev><<<grid, block, roi_w * 3 * 2 + 16 * 2, stream>>>( \
                       in_buf, out_buf, h_in, w_in, h_out, w_out, img_h, img_w, pad_h, pad_w, h_scale, w_scale, \
                       roi_h_start, roi_w_start, roi_h, roi_w, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3); \
            break; \
        default: \
            grid.x = is_bilinear ? h_out / (1 << ALIGN1B) : grid.x; \
            rgb_resize_ROI_norm_kernel<is_bilinear, align_in_w, 1 << ALIGN1B, channel_rev><<<grid, block, roi_w * 3 * 2 + 16 * 2, stream>>>( \
                       in_buf, out_buf, h_in, w_in, h_out, w_out, img_h, img_w, pad_h, pad_w, h_scale, w_scale, \
                       roi_h_start, roi_w_start, roi_h, roi_w, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3); \
            break; \
    }

void RGBROIBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, bool channel_rev,
        cudaStream_t stream) {
    int img_h = h_out;
    int img_w = w_out;
    float w_scale = 1.0f * roi_w / img_w;
    float h_scale = 1.0f * roi_h / img_h; 
    int pad_h = 0;
    int pad_w = 0;
    float pad1 = 0.f;
    float pad2 = 0.f;
    float pad3 = 0.f;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0 && (w_in & 7) == 0) {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, true, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, true, 0);
        }
    } else {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, false, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, false, 0);
        }
    }
}

void RGBROINearestResizeNormPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, bool channel_rev,
        cudaStream_t stream) {

    int img_h = h_out;
    int img_w = w_out;
    float w_scale = 1.0f * roi_w / img_w;
    float h_scale = 1.0f * roi_h / img_h; 
    int pad_h = 0;
    int pad_w = 0;
    float pad1 = 0.f;
    float pad2 = 0.f;
    float pad3 = 0.f;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0 && (w_in & 7) == 0) {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, true, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, true, 0);
        }
    } else {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, false, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, false, 0);
        }
    }
}

void RGBROIBilinearResizeNormPadPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out, int img_w, int img_h, int pad_w, int pad_h,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float pad1, float pad2, float pad3, bool channel_rev,
        cudaStream_t stream) {
    float w_scale = 1.0f * roi_w / img_w;
    float h_scale = 1.0f * roi_h / img_h; 
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0 && (w_in & 7) == 0) {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, true, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, true, 0);
        }
    } else {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, false, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(true, false, 0);
        }
    }
}

void RGBROINearestResizeNormPadPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out, int img_w, int img_h, int pad_w, int pad_h,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float pad1, float pad2, float pad3, bool channel_rev,
        cudaStream_t stream) {
    float w_scale = 1.0f * roi_w / img_w;
    float h_scale = 1.0f * roi_h / img_h; 
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0 && (w_in & 7) == 0) {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, true, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, true, 0);
        }
    } else {
        if (channel_rev) {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, false, 2);
        } else {
            LAUNCH_RGB_RESIZE_NORM_PLANE_KERNEL(false, false, 0);
        }
    }
}

void BatchRGBBilinearResizeNormPadPlane(uint8_t *in_buf, float *out_buf, int batch,
        int input_batch_offset, int output_batch_offset, batch_resize_param * param,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float pad1, float pad2, float pad3,
        bool channel_rev, cudaStream_t stream) {
    dim3 block(512, 1, 1);
    dim3 grid(batch, 1, 1);
    if (channel_rev) {
        batch_rgb_resize_norm_kernel<true><<<grid, block, 0, stream>>>(in_buf, out_buf, input_batch_offset, output_batch_offset,
                param, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3);
    } else {
        batch_rgb_resize_norm_kernel<false><<<grid, block, 0, stream>>>(in_buf, out_buf, input_batch_offset, output_batch_offset,
                param, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3);
    }
}

void MultiRoiRGBBilinearResizeNormPadPlane(uint8_t *in_buf, float *out_buf, int batch,
                                           int in_h, int in_w, int output_batch_offset, multi_roi_resize_param *param,
                                           float scale, float mean1, float mean2, float mean3,
                                           float std1, float std2, float std3, float pad1, float pad2, float pad3,
                                           bool channel_rev, cudaStream_t stream)
{
    dim3 block(256, 1, 1);
    dim3 grid(16, batch, 1);
    if (channel_rev)
    {
        multi_roi_rgb_resize_norm_kernel<true><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_h, in_w, output_batch_offset,
            param, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3);
    }
    else
    {
        multi_roi_rgb_resize_norm_kernel<false><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_h, in_w, output_batch_offset,
            param, scale, mean1, mean2, mean3, std1, std2, std3, pad1, pad2, pad3);
    }
}
#include "cuda.h"
#include "yuv2rgb_resize_norm.h"

void NV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_resize_norm_fuse_kernel<false><<<grid, block, in_w * 4 + 8*2, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_resize_norm_fuse_kernel<false><<<grid, block, in_w * 4 + 8*3, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void NV12ToRGBNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_nearest_resize_norm_fuse_kernel<false><<<grid, block, in_w * 2 + 8*2, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToRGBNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_nearest_resize_norm_fuse_kernel<false><<<grid, block, in_w * 2 + 8*3, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void NV12ToBGRBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_resize_norm_fuse_kernel<true><<<grid, block, in_w * 4 + 8*2, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToBGRBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_resize_norm_fuse_kernel<true><<<grid, block, in_w * 4 + 8*3, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void NV12ToBGRNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_nearest_resize_norm_fuse_kernel<true><<<grid, block, in_w * 2 + 8*2, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToBGRNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_nearest_resize_norm_fuse_kernel<true><<<grid, block, in_w * 2 + 8*3, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiNV12ToRGBBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 4 + 8*6;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, true, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, true, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, false, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, false, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiYU12ToRGBBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 4 + 8*6;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, true, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, true, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, false, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, false, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiNV12ToBGRBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 4 + 8*6;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_nv122rgb_resize_norm_fuse_kernel<true, false, true, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_nv122rgb_resize_norm_fuse_kernel<true, false, true, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_nv122rgb_resize_norm_fuse_kernel<true, false, false, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_nv122rgb_resize_norm_fuse_kernel<true, false, false, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiYU12ToBGRBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 4 + 8*6;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_yu122rgb_resize_norm_fuse_kernel<true, false, true, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yu122rgb_resize_norm_fuse_kernel<true, false, true, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yu122rgb_resize_norm_fuse_kernel<true, false, false, true, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yu122rgb_resize_norm_fuse_kernel<true, false, false, false, false/*no norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

#define ROI_YUV2RGB_RESIZE_QUANTIZE_KERNEL(kernel, input_align)\
{\
    dim3 block(256,1,1);\
    dim3 grid(out_h, 1, 1);\
    switch (out_w % 4) {\
        case 0:\
            if (out_w / 4 < 512) block.x = out_w / 4;\
            kernel<false, false, input_align, ALIGN4B, false><<<grid, block, sm_size, stream>>>(\
                   ws, out_buf, roi_w, roi_h,  img_w, img_h, out_w, out_h, pad_w, pad_h,\
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3, zero_point, 1.f /scales_input);\
            break;\
        case 2:\
            if (out_w / 2 < 512) block.x = out_w / 2;\
            kernel<false, false, input_align, ALIGN2B, false><<<grid, block, sm_size, stream>>>(\
                   ws, out_buf, roi_w, roi_h,  img_w, img_h, out_w, out_h, pad_w, pad_h,\
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3, zero_point, 1.f /scales_input);\
            break;\
        default:\
            kernel<false, false, input_align, ALIGN1B, false><<<grid, block, sm_size, stream>>>(\
                   ws, out_buf, roi_w, roi_h,  img_w, img_h, out_w, out_h, pad_w, pad_h,\
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3, zero_point, 1.f /scales_input);\
    }\
}

void RoiNV12ToRGBBilinearResizeQuantizePlane(uint8_t *in_buf, uint8_t *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, float zero_point, float scales_input, cudaStream_t stream) {
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 2 + 8*3;
    dim3 block(256,1,1);
    dim3 grid(roi_h, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
        roi_nv122rgba_kernel<false, false, true><<<grid, block, sm_size, stream>>>(in_buf, ws,
                in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h);
    } else {
        roi_nv122rgba_kernel<false, false, false><<<grid, block, sm_size, stream>>>(in_buf, ws,
                in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h);
    }
    sm_size = roi_w * 8 + 8;
    if (roi_w % 2 == 0) {
        ROI_YUV2RGB_RESIZE_QUANTIZE_KERNEL(rgba_resize_norm_quantize_fuse_kernel, true);
    } else {
        ROI_YUV2RGB_RESIZE_QUANTIZE_KERNEL(rgba_resize_norm_quantize_fuse_kernel, false);
    }
}

void RoiYU12ToRGBBilinearResizeQuantizePlane(uint8_t *in_buf, uint8_t *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, float zero_point, float scales_input, cudaStream_t stream) {
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 2 + 8*3;
    dim3 block(256,1,1);
    dim3 grid(roi_h, 1, 1);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
        roi_yu122rgba_kernel<false, false, true><<<grid, block, sm_size, stream>>>(in_buf, ws,
                in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h);
    } else {
        roi_yu122rgba_kernel<false, false, false><<<grid, block, sm_size, stream>>>(in_buf, ws,
                in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h);
    }
    sm_size = roi_w * 8 + 8;
    if (roi_w % 2 == 0) {
        ROI_YUV2RGB_RESIZE_QUANTIZE_KERNEL(rgba_resize_norm_quantize_fuse_kernel, true);
    } else {
        ROI_YUV2RGB_RESIZE_QUANTIZE_KERNEL(rgba_resize_norm_quantize_fuse_kernel, false);
    }
}

void RoiNV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 4 + 8*6;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, true, true, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, true, false, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, false, true, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_nv122rgb_resize_norm_fuse_kernel<false, false, false, false, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiYU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 4 + 8*6;
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, true, true, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, true, false, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, false, true, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yu122rgb_resize_norm_fuse_kernel<false, false, false, false, true/*norm*/><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void FullRangeNV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_resize_norm_fuse_kernel<false, true><<<grid, block, in_w * 4 + 8*2, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void FullRangeYU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_resize_norm_fuse_kernel<false, true><<<grid, block, in_w * 4 + 8*3, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYUV444PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 6 + 8*6;
    if ((out_w & 1) == 0) sm_size += out_w * 3 * sizeof(float);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_yuv444p2rgb_resize_norm_fuse_kernel<false, false, true, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yuv444p2rgb_resize_norm_fuse_kernel<false, false, true, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yuv444p2rgb_resize_norm_fuse_kernel<false, false, false, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yuv444p2rgb_resize_norm_fuse_kernel<false, false, false, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiYUV400PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean, float std, float scale, float pad, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 2 + 8*2;
    if ((out_w & 1) == 0) sm_size += out_w * sizeof(float);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_yuv400p2rgb_resize_norm_fuse_kernel<true, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean, std, scale, y_ratio, x_ratio, pad);
      } else {
          roi_yuv400p2rgb_resize_norm_fuse_kernel<true, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean, std, scale, y_ratio, x_ratio, pad);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yuv400p2rgb_resize_norm_fuse_kernel<false, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean, std, scale, y_ratio, x_ratio, pad);
      } else {
          roi_yuv400p2rgb_resize_norm_fuse_kernel<false, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean, std, scale, y_ratio, x_ratio, pad);
      }
    }
}

void RoiYUV422PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = roi_w * 6 + 8*6;
    if ((out_w & 1) == 0) sm_size += out_w * 3 * sizeof(float);
    if (((uint64_t)in_buf & 7) == 0 && (roi_w & 7) == 0 && (roi_w_start & 7) == 0) {
      if ((out_w & 1) == 0) {
          roi_yuv422p2rgb_resize_norm_fuse_kernel<false, false, true, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yuv422p2rgb_resize_norm_fuse_kernel<false, false, true, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yuv422p2rgb_resize_norm_fuse_kernel<false, false, false, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yuv422p2rgb_resize_norm_fuse_kernel<false, false, false, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiYUV422ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid(1, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    int sm_size = (roi_w + 2) * 2 * 2 + 8*2;
    if ((out_w & 1) == 0) sm_size += out_w * 3 * sizeof(float);
    if (((uint64_t)in_buf & 3) == 0 && (roi_w & 3) == 0 && (roi_w_start & 3) == 0) {
      if ((out_w & 1) == 0) {
          roi_yuv4222rgb_resize_norm_fuse_kernel<false, false, true, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yuv4222rgb_resize_norm_fuse_kernel<false, false, true, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    } else {
      if ((out_w & 1) == 0) {
          roi_yuv4222rgb_resize_norm_fuse_kernel<false, false, false, true><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      } else {
          roi_yuv4222rgb_resize_norm_fuse_kernel<false, false, false, false><<<grid, block, sm_size, stream>>>(
                  in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
                  mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
      }
    }
}

void RoiYU12ToRGBBilinearResizeNormPlaneV2(uint8_t *in_buf, float *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, uint8_t pad1, uint8_t pad2, uint8_t pad3, cudaStream_t stream){
    int pad_img_w = roi_w + pad_w * 2;
    int pad_img_h = roi_h + pad_h * 2;
    float horiz_filterscale = pad_img_w * 1.f / out_w;
    if (horiz_filterscale < 1.0) {
    	horiz_filterscale = 1.0;
    }
    /* maximum number of coeffs */
    int horiz_ksize = (int)ceil(horiz_filterscale) * 2 + 1;


    float vert_filterscale = pad_img_h * 1.f / out_h;
    if (vert_filterscale < 1.0) {
    	vert_filterscale = 1.0;
    }
    /* maximum number of coeffs */
    int vert_ksize = (int)ceil(vert_filterscale) * 2 + 1;

    int ws_h = pad_img_h;
    int ws_w = out_w;
    
    uchar4 *rgb_buf = ws;
    uchar4 *ws_buf = rgb_buf + pad_img_w * pad_img_h;
    dim3 cvt_block(256, 1, 1);
    dim3 cvt_grid(pad_img_h, 1, 1);
    int sm_size = roi_w * 2 + 8*6;
    roi_yu122rgba_pad_kernel<false, false, false><<<cvt_grid, cvt_block, sm_size>>>(in_buf, rgb_buf, in_w, in_h, 
        roi_w_start, roi_h_start, roi_w, roi_h, pad_img_w, pad_img_h, pad_w, pad_h, pad1, pad2, pad3);

    dim3 hori_block(256,1,1);
    if (ws_w < 512) hori_block.x = ws_w;
    dim3 hori_grid(1, ws_h, 1);
    int hori_sm_size = 0;
    switch (horiz_ksize) {
    case 3:
        rgba_horizontal_resize_kernel<3/*KSIZE*/><<<hori_grid, hori_block, hori_sm_size, stream>>>(
                rgb_buf, ws_buf, pad_img_w, pad_img_h, ws_w, ws_h, horiz_filterscale);
		break;
    case 5:
        rgba_horizontal_resize_kernel<5/*KSIZE*/><<<hori_grid, hori_block, hori_sm_size, stream>>>(
                rgb_buf, ws_buf, pad_img_w, pad_img_h, ws_w, ws_h, horiz_filterscale);
    	break;
    default:
    	printf("roi_yu122rgb_horizontal_resize_fuse_kernel not support kernel size %d\n", horiz_ksize);
    	abort();
    }

    dim3 vert_block(256,1,1);
    if (out_w < 512) vert_block.x = out_w;
    dim3 vert_grid(1, out_h, 1);
    int vert_sm_size = 0;
    switch (vert_ksize) {
    case 3:
		rgba_vertical_resize_kernel<false, false, true/*norm*/, 3/*KSIZE*/><<<vert_grid, vert_block, vert_sm_size, stream>>>(
				ws_buf, out_buf, ws_w, ws_h, out_w, out_h, vert_filterscale,  mean1, mean2, mean3, std1, std2, std3, scale);
		break;
    case 5:
		rgba_vertical_resize_kernel<false, false, true/*norm*/, 5/*KSIZE*/><<<vert_grid, vert_block, vert_sm_size, stream>>>(
				ws_buf, out_buf, ws_w, ws_h, out_w, out_h, vert_filterscale,  mean1, mean2, mean3, std1, std2, std3, scale);
		break;
    default:
    	printf("roi_yu122rgb_vertical_resize_fuse_kernel not support kernel size %d\n", vert_ksize);
    	abort();
    }
}

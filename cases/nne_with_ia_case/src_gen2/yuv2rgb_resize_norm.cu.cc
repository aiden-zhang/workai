#include "cuda.h"
#include "yuv2rgb_resize_norm.h"

void NV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_resize_norm_fuse_kernel<false><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_resize_norm_fuse_kernel<false><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void NV12ToRGBNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_nearest_resize_norm_fuse_kernel<false><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToRGBNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_nearest_resize_norm_fuse_kernel<false><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void NV12ToBGRBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_resize_norm_fuse_kernel<true><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToBGRBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_resize_norm_fuse_kernel<true><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void NV12ToBGRNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_nearest_resize_norm_fuse_kernel<true><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void YU12ToBGRNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_nearest_resize_norm_fuse_kernel<true><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiNV12ToRGBBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_nv122rgb_resize_norm_fuse_kernel<false, false, false/*no norm*/><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYU12ToRGBBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w * out_h + block.x - 1) / block.x, 1, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yu122rgb_resize_norm_general_kernel<false, false, false/*norm*/><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiNV12ToBGRBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_nv122rgb_resize_norm_fuse_kernel<true, false, false/*no norm*/><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYU12ToBGRBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w * out_h + block.x - 1) / block.x, 1, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yu122rgb_resize_norm_general_kernel<true, false, false/*norm*/><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiNV12ToRGBBilinearResizeQuantizePlane(uint8_t *in_buf, uint8_t *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, float zero_point, float scales_input, cudaStream_t stream) {
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
     dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((roi_w + block.x - 1) / block.x, roi_h, 1);
    roi_nv122rgba_kernel<false><<<grid, block, 0, stream>>>(in_buf, ws,
            in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h);
    grid.x = (out_w + block.x - 1) / block.x;
    grid.y = out_h;
    rgba_resize_norm_quantize_fuse_kernel<false, false><<<grid, block, 0, stream>>>(
            ws, out_buf, roi_w, roi_h,  img_w, img_h, out_w, out_h, pad_w, pad_h,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3, zero_point, 1.f / scales_input);
}

void RoiYU12ToRGBBilinearResizeQuantizePlane(uint8_t *in_buf, uint8_t *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, float zero_point, float scales_input, cudaStream_t stream) {
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((roi_w + block.x - 1) / block.x, roi_h, 1);
    roi_yu122rgba_kernel<false><<<grid, block, 0, stream>>>(in_buf, ws,
            in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h);
    grid.x = (out_w + block.x - 1) / block.x;
    grid.y = out_h;
    rgba_resize_norm_quantize_fuse_kernel<false, false><<<grid, block, 0, stream>>>(
            ws, out_buf, roi_w, roi_h,  img_w, img_h, out_w, out_h, pad_w, pad_h,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, y_ratio, x_ratio, pad1, pad2, pad3, zero_point, 1.f / scales_input);
}

void RoiNV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_nv122rgb_resize_norm_fuse_kernel<false, false, true/*norm*/><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w * out_h + block.x - 1) / block.x, 1, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yu122rgb_resize_norm_general_kernel<false, false, true/*norm*/><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void FullRangeNV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    nv122rgb_resize_norm_fuse_kernel<false, true><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void FullRangeYU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 768) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = in_w * 1.f / img_w;
    float y_ratio = in_h * 1.f / img_h;
    yu122rgb_resize_norm_fuse_kernel<false, true><<<grid, block, 0, stream>>>(
        in_buf, out_buf, in_w, in_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
        mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYUV444PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w * out_h + block.x - 1) / block.x, 1, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yuv444p2rgb_resize_norm_general_kernel<false, false><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYUV400PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean, float std, float scale, float pad, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w * out_h + block.x - 1) / block.x, 1, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yuv400p2rgb_resize_norm_general_kernel<<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            mean, std, scale, y_ratio, x_ratio, pad);
}

void RoiYUV422PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w * out_h + block.x - 1) / block.x, 1, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yuv422p2rgb_resize_norm_general_kernel<false, false><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}

void RoiYUV422ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream) {
    dim3 block(256,1,1);
    if (out_w < 512) block.x = out_w;
    dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    float x_ratio = roi_w * 1.f / img_w;
    float y_ratio = roi_h * 1.f / img_h;
    roi_yuv4222rgb_resize_norm_fuse_kernel<false, false><<<grid, block, 0, stream>>>(
            in_buf, out_buf, in_w, in_h, roi_w_start, roi_h_start, roi_w, roi_h, img_w, img_h, out_w, out_h, pad_w, pad_h,
            mean1, mean2, mean3, std1, std2, std3, scale, y_ratio, x_ratio, pad1, pad2, pad3);
}


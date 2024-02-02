#include "cuda.h"
#include "pytorch_resize_norm.h"


#define MAX_KSIZE 51

void launch_rgba_hori_resize(uchar4* in, uchar4* out, int in_w, int in_h, int out_w, int out_h, 
        float filter_scale, int ksize, cudaStream_t stream, Int2Type<1>) {
}

void launch_rgba_vert_resize(uchar4* in, float* out, int in_w, int in_h, int out_w, int out_h,
        float filter_scale, float mean1, float mean2, float mean3, float std1, float std2,
        float std3, float scale, int ksize, cudaStream_t stream, Int2Type<1>) {
}

template <int KSIZE>
void launch_rgba_hori_resize(uchar4* in, uchar4* out, int in_w, int in_h, int out_w, int out_h, 
        float filter_scale, int ksize, cudaStream_t stream, Int2Type<KSIZE>) {
    if (ksize == KSIZE) {
        dim3 hori_block(256, 1, 1);
        if (out_w < 512) hori_block.x = out_w;
        dim3 hori_grid(1, out_h, 1);
        rgba_horizontal_resize_kernel<KSIZE><<<hori_grid, hori_block, 0, stream>>>(
                in, out, in_w, in_h, out_w, out_h, filter_scale);
    } else {
        launch_rgba_hori_resize(in, out, in_w, in_h, out_w, out_h, filter_scale, ksize, stream, Int2Type<KSIZE - 2>());
    }
}

template <int KSIZE>
void launch_rgba_vert_resize(uchar4* in, float* out, int in_w, int in_h, int out_w, int out_h,
        float filter_scale, float mean1, float mean2, float mean3, float std1, float std2,
        float std3, float scale, int ksize, cudaStream_t stream, Int2Type<KSIZE>) {
    if (ksize == KSIZE) {
        dim3 vert_block(256, 1, 1);
        if (out_w < 512) vert_block.x = out_w;
        dim3 vert_grid(1, out_h, 1);
        rgba_vertical_resize_kernel<false, true/*norm*/, KSIZE><<<vert_grid, vert_block, 0, stream>>>(
                in, out, in_w, in_h, out_w, out_h, filter_scale, mean1, mean2, mean3, std1, std2, std3, scale);
    } else {
        launch_rgba_vert_resize(in, out, in_w, in_h, out_w, out_h,
                filter_scale, mean1, mean2, mean3, std1, std2, std3, scale, ksize, stream, Int2Type<KSIZE - 2>());
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
    dim3 cvt_grid((pad_img_w + cvt_block.x - 1) / cvt_block.x, pad_img_h, 1);
    roi_yu122rgba_pad_kernel<false><<<cvt_grid, cvt_block, 0, stream>>>(in_buf, rgb_buf, in_w, in_h, 
        roi_w_start, roi_h_start, roi_w, roi_h, pad_img_w, pad_img_h, pad_w, pad_h, pad1, pad2, pad3);

    launch_rgba_hori_resize(rgb_buf, ws_buf, pad_img_w, pad_img_h, ws_w, ws_h, horiz_filterscale, horiz_ksize, stream, Int2Type<MAX_KSIZE>());

    launch_rgba_vert_resize(ws_buf, out_buf, ws_w, ws_h, out_w, out_h,
            vert_filterscale, mean1, mean2, mean3, std1, std2, std3, scale, vert_ksize, stream, Int2Type<MAX_KSIZE>());

}

void launch_rgba_hori_resize(uint8_t* in, uchar4* out, int in_w, int in_h, int out_w, int out_h, 
        float filter_scale, int ksize, cudaStream_t stream, Int2Type<1>) {
}

template <int KSIZE>
void launch_rgba_hori_resize(uint8_t* in, uchar4* out, int in_w, int in_h, int out_w, int out_h, 
        float filter_scale, int ksize, cudaStream_t stream, Int2Type<KSIZE>) {
    if (ksize == KSIZE) {
        dim3 hori_block(256, 1, 1);
        dim3 hori_grid(1, out_h, 1);
        rgba_horizontal_resize_kernel<KSIZE><<<hori_grid, hori_block, 0, stream>>>(
                in, out, in_w, in_h, out_w, out_h, filter_scale);
    } else {
        launch_rgba_hori_resize(in, out, in_w, in_h, out_w, out_h, filter_scale, ksize, stream, Int2Type<KSIZE - 2>());
    }
}

void launch_rgba_vertical_resize_crop_norm(uchar4* in, float* out,
        int in_w, int in_h, int out_w, int out_h, int crop_start_w, int crop_start_h,
        float filter_scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale, bool fmt_cvt, int ksize,
        cudaStream_t stream, Int2Type<1>) {
}

template <int KSIZE>
void launch_rgba_vertical_resize_crop_norm(uchar4* in, float* out,
        int in_w, int in_h, int out_w, int out_h, int crop_start_w, int crop_start_h,
        float filter_scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale, bool fmt_cvt, int ksize,
        cudaStream_t stream, Int2Type<KSIZE>) {
    if (ksize == KSIZE) {
        dim3 vert_block(256, 1, 1);
        if (out_w < 512) vert_block.x = out_w;
        dim3 vert_grid(1, out_h, 1);
        if(fmt_cvt) {
            rgba_vertical_resize_crop_norm_kernel<true, true/*norm*/, KSIZE><<<vert_grid, vert_block, 0, stream>>>(
                in, out, in_w, in_h, out_w, out_h, crop_start_w, crop_start_h, filter_scale, mean1, mean2, mean3,
                std1, std2, std3, scale);
        } else {
            rgba_vertical_resize_crop_norm_kernel<false, true/*norm*/, KSIZE><<<vert_grid, vert_block, 0, stream>>>(
                in, out, in_w, in_h, out_w, out_h, crop_start_w, crop_start_h, filter_scale, mean1, mean2, mean3,
                std1, std2, std3, scale);
        }
    } else {
        launch_rgba_vertical_resize_crop_norm(in, out, in_w, in_h, out_w, out_h, crop_start_w, crop_start_h,
                filter_scale, mean1, mean2, mean3, std1, std2, std3, scale, fmt_cvt, ksize, stream, Int2Type<KSIZE - 2>());
    }
}



void RGBBilinearResizeCropNormPlaneV2(uint8_t *in_buf, float *out_buf, uchar4 *ws, int in_w, int in_h,
    int resized_w, int resized_h, int crop_w_start, int crop_h_start, int crop_w, int crop_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, bool fmt_cvt, cudaStream_t stream){
    float horiz_filterscale = in_w * 1.f / resized_w;
    if (horiz_filterscale < 1.0) {
      horiz_filterscale = 1.0;
    }
    /* maximum number of coeffs */
    int horiz_ksize = (int)ceil(horiz_filterscale) * 2 + 1;


    float vert_filterscale = in_h * 1.f / resized_h;
    if (vert_filterscale < 1.0) {
      vert_filterscale = 1.0;
    }
    /* maximum number of coeffs */
    int vert_ksize = (int)ceil(vert_filterscale) * 2 + 1;

    int ws_h = in_h;
    int ws_w = resized_w;
    
    uchar4 *rgba_buf = ws;
    launch_rgba_hori_resize(in_buf, rgba_buf, in_w, in_h, ws_w, ws_h, horiz_filterscale, horiz_ksize,
            stream, Int2Type<MAX_KSIZE>());

    launch_rgba_vertical_resize_crop_norm(rgba_buf, out_buf, ws_w, ws_h, crop_w, crop_h,
            crop_w_start, crop_h_start, vert_filterscale, mean1, mean2, mean3, std1, std2, std3, scale,
            fmt_cvt, vert_ksize, stream, Int2Type<MAX_KSIZE>());

}

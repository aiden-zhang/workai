#ifndef YUV2RGB_RESIZE_NORM_H_
#define YUV2RGB_RESIZE_NORM_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"


__device__ __forceinline__ float clip(float value) {
  value += 0.5f;
  value = min(max(0.f,value),255.f);
  return value;
}

__device__ __forceinline__ float3 convert2rgb_TV_range(float y, float u, float v) {
    float3 tmp;
    y -= 16.f;
    y = max(0.f, y);
    u -= 128.f;
    v -= 128.f;
    tmp.x = clip(1.164f * y + 1.596f * v);
    tmp.y = clip(1.164f * y - 0.813f * v - 0.391f * u);
    tmp.z = clip(1.164f * y + 2.018f * u);
    return tmp;
}

__device__ __forceinline__ float3 convert2rgb_full_range(float y, float u, float v) {
    float3 tmp;
    u -= 128.f;
    v -= 128.f;
    tmp.x = clip(y + 1.403f * v);
    tmp.y = clip(y - 0.344f * u - 0.714f * v);
    tmp.z = clip(y + 1.773f * u);
    return tmp;
}


template <bool bgr_format = false, bool full_range = false>
__global__ void nv122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y = sm;
    uint8_t *uv = sm + (in_w * 2 / 8 + 1) * 8;
    int h_idx = blockIdx.y; 
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
        int top_h = raw_h_idx;
        int sm_offset_y = 0, sm_offset_uv = 0;
        int line_id_y  =  top_h + 1 < in_h ? 1 : 0;
        int line_id_uv = (top_h & 0x1) == 0 ? 0 : 1;
        int next_line_y  = in_w * line_id_y;
        int next_line_uv = in_w * line_id_uv;
        float y_diff = raw_h_idx - top_h;
        if ( h_run) {
            int copy_size = top_h + 1 < in_h ? 2 * in_w : in_w;
            sm_offset_y = global2share_copy(in + top_h * in_w , y, copy_size);
            copy_size = (top_h & 0x1) == 0 ? in_w : 2 * in_w;
            sm_offset_uv = global2share_copy(in + in_w * in_h + (top_h >> 1) * in_w , uv, copy_size);
            y += sm_offset_y;
            uv += sm_offset_uv;
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(in_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                uchar2 a_uv = ((uchar2*)(uv))[left_w>>1];
                uchar2 b_uv = ((uchar2*)(uv))[right_w>>1];
                uchar2 c_uv = ((uchar2*)(uv + next_line_uv))[left_w>>1];
                uchar2 d_uv = ((uchar2*)(uv + next_line_uv))[right_w>>1];
                float3 a, b, c, d;

                float a_y = y[left_w];
                float b_y = y[right_w];
                float c_y = y[next_line_y + left_w];
                float d_y = y[next_line_y + right_w];
                if (full_range) {
                    a = convert2rgb_full_range(a_y, a_uv.x, a_uv.y);
                    b = convert2rgb_full_range(b_y, b_uv.x, b_uv.y);
                    c = convert2rgb_full_range(c_y, c_uv.x, c_uv.y);
                    d = convert2rgb_full_range(d_y, d_uv.x, d_uv.y);
                } else {
                    a = convert2rgb_TV_range(a_y, a_uv.x, a_uv.y);
                    b = convert2rgb_TV_range(b_y, b_uv.x, b_uv.y);
                    c = convert2rgb_TV_range(c_y, c_uv.x, c_uv.y);
                    d = convert2rgb_TV_range(d_y, d_uv.x, d_uv.y);
                }
                uchar3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            out[out_idx] = norm_tmp.x;
            out[out_idx + out_w * out_h] = norm_tmp.y;
            out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
        }
    }
}

template <bool bgr_format = false, bool full_range = false>
__global__ void yu122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y = sm;
    uint8_t *u = sm + (in_w * 2 / 8 + 1) * 8;
    uint8_t *v = sm + (in_w * 2 / 8 + 1) * 8 + (in_w * 2 / 16 + 1) * 8;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
        int top_h = raw_h_idx;
        int sm_offset_y = 0, sm_offset_u = 0, sm_offset_v = 0;
        int line_id_y  =  top_h + 1 < in_h ? 1 : 0;
        int line_id_uv = (top_h & 0x1) == 0 ? 0 : 1;
        int next_line_y  = in_w * line_id_y;
        int next_line_uv = (in_w>>1) * line_id_uv;
        float y_diff = raw_h_idx - top_h;
        if (h_run) {
            int copy_size = top_h + 1 < in_h ? 2 * in_w : in_w;
            sm_offset_y = global2share_copy(in + top_h * in_w , y, copy_size);
            copy_size = (top_h & 0x1) == 0 ? (in_w >> 1) : in_w;
            sm_offset_u = global2share_copy(in + in_w * in_h + (top_h >> 1) * (in_w >> 1), u, copy_size);
            sm_offset_v = global2share_copy(in + in_w * in_h + (in_w * in_h >> 2) + (top_h >> 1) * (in_w >> 1), v, copy_size);
            y += sm_offset_y;
            u += sm_offset_u;
            v += sm_offset_v;
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(in_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                float a_y = y[left_w];
                float b_y = y[right_w];
                float c_y = y[next_line_y + left_w];
                float d_y = y[next_line_y + right_w];
                float2 a_uv, b_uv, c_uv, d_uv;
                a_uv.x = u[left_w>>1];
                a_uv.y = v[left_w>>1];
                b_uv.x = u[right_w>>1];
                b_uv.y = v[right_w>>1];
                c_uv.x = u[next_line_uv + (left_w>>1)];
                c_uv.y = v[next_line_uv + (left_w>>1)];
                d_uv.x = u[next_line_uv + (right_w>>1)];
                d_uv.y = v[next_line_uv + (right_w>>1)];
                float3 a, b, c, d;
                if (full_range) {
                    a = convert2rgb_full_range(a_y, a_uv.x, a_uv.y);
                    b = convert2rgb_full_range(b_y, b_uv.x, b_uv.y);
                    c = convert2rgb_full_range(c_y, c_uv.x, c_uv.y);
                    d = convert2rgb_full_range(d_y, d_uv.x, d_uv.y);
                } else {
                    a = convert2rgb_TV_range(a_y, a_uv.x, a_uv.y);
                    b = convert2rgb_TV_range(b_y, b_uv.x, b_uv.y);
                    c = convert2rgb_TV_range(c_y, c_uv.x, c_uv.y);
                    d = convert2rgb_TV_range(d_y, d_uv.x, d_uv.y);
                }

                uchar3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            out[out_idx] = norm_tmp.x;
            out[out_idx + out_w * out_h] = norm_tmp.y;
            out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
        }
    }
}


template <bool bgr_format = false, bool full_range = false>
__global__ void nv122rgb_nearest_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y = sm;
    uint8_t *uv = sm + ((in_w + 8) & 0xfffffff8);
    int h_idx = blockIdx.y; 
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        int raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
        if ( h_run) {
            int offset = 0;
            offset = global2share_copy(in + raw_h_idx * in_w , y, in_w);
            y += offset;
            offset = global2share_copy(in + in_w * in_h + (raw_h_idx >> 1) * in_w , uv, in_w);
            uv += offset;
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            float3 norm_tmp = {pad1, pad2, pad3};
            if (h_run && w_run) {
                int raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                uchar2 a_uv = ((uchar2*)(uv))[raw_w_idx>>1];
                float a_y = y[raw_w_idx];
                float3 out_tmp;
                if (full_range) {
                    out_tmp = convert2rgb_full_range(a_y, a_uv.x, a_uv.y);
                } else {
                    out_tmp = convert2rgb_TV_range(a_y, a_uv.x, a_uv.y);
                }
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            out[out_idx] = norm_tmp.x;
            out[out_idx + out_w * out_h] = norm_tmp.y;
            out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
        }
    }
}

template <bool bgr_format = false, bool full_range = false>
__global__ void yu122rgb_nearest_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y = sm;
    uint8_t *u = sm + (in_w / 8 + 1) * 8;
    uint8_t *v = sm + (in_w / 8 + 1) * 8 + (in_w / 16 + 1) * 8;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        int raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
        if (h_run) {
            int offset = 0;
            offset = global2share_copy(in + raw_h_idx * in_w , y, in_w);
            y += offset;
            offset = global2share_copy(in + in_w * in_h + (raw_h_idx >> 1) * (in_w >> 1), u, in_w >> 1);
            u += offset;
            offset = global2share_copy(in + in_w * in_h + (in_w * in_h >> 2) + (raw_h_idx >> 1) * (in_w >> 1), v, in_w >> 1);
            v += offset;
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                int raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                float a_y = y[raw_w_idx];
                float2 a_uv;
                a_uv.x = u[raw_w_idx>>1];
                a_uv.y = v[raw_w_idx>>1];
                float3 out_tmp;
                if (full_range) {
                    out_tmp = convert2rgb_full_range(a_y, a_uv.x, a_uv.y);
                } else {
                    out_tmp = convert2rgb_TV_range(a_y, a_uv.x, a_uv.y);
                }
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            out[out_idx] = norm_tmp.x;
            out[out_idx + out_w * out_h] = norm_tmp.y;
            out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
        }
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, bool out_align = false, bool norm = true>
__global__ void roi_nv122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *uv1 = y2 + (roi_w & 0xfffffff8) + 8;
    uint8_t *uv2 = uv1 + (roi_w & 0xfffffff8) + 8;
    int h_idx = blockIdx.y; 
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
        int top_h = raw_h_idx;
        int bottom_h = MIN(in_h - 1, top_h + 1);
        float y_diff = raw_h_idx - top_h;
        if ( h_run) {
          if (align) {
            global2share_copy_align(in + top_h * in_w + roi_w_start, y1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) & 0xfffffffe;
            global2share_copy_align(in + in_w * in_h + (top_h >> 1) * in_w + (roi_w_start & 0xfffffffe), uv1, copy_size);
            global2share_copy_align(in + in_w * in_h + (bottom_h >> 1) * in_w + (roi_w_start & 0xfffffffe), uv2, copy_size);
          } else {
            int sm_offset = global2share_copy(in + top_h * in_w + roi_w_start, y1, roi_w);
            y1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            y2 += sm_offset;
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) & 0xfffffffe;
            sm_offset = global2share_copy(in + in_w * in_h + (top_h >> 1) * in_w + (roi_w_start & 0xfffffffe), uv1, copy_size);
            uv1 += sm_offset;
            sm_offset = global2share_copy(in + in_w * in_h + (bottom_h >> 1) * in_w + (roi_w_start & 0xfffffffe), uv2, copy_size);
            uv2 += sm_offset;
          }
        }
        __syncthreads();

        int num_v = 1;
        if (out_align) {
          out_w = out_w >> 1;
          num_v = 2;
        }
        for (int tw_idx = threadIdx.x; tw_idx < out_w; tw_idx += blockDim.x) {
          float t1[2], t2[2], t3[2];
          for (int v_idx = 0; v_idx < num_v; v_idx++) {
            int w_idx = tw_idx * num_v + v_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(roi_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                int left = (left_w + (roi_w_start & 1))>>1;
                int right = (right_w + (roi_w_start & 1))>>1;
                uchar2 a_uv = ((uchar2*)uv1)[left];
                uchar2 b_uv = ((uchar2*)uv1)[right];
                uchar2 c_uv = ((uchar2*)uv2)[left];
                uchar2 d_uv = ((uchar2*)uv2)[right];
                float3 a, b, c, d;

                float a_y = y1[left_w];
                float b_y = y1[right_w];
                float c_y = y2[left_w];
                float d_y = y2[right_w];
                if (full_range) {
                    a = convert2rgb_full_range(a_y, a_uv.x, a_uv.y);
                    b = convert2rgb_full_range(b_y, b_uv.x, b_uv.y);
                    c = convert2rgb_full_range(c_y, c_uv.x, c_uv.y);
                    d = convert2rgb_full_range(d_y, d_uv.x, d_uv.y);
                } else {
                    a = convert2rgb_TV_range(a_y, a_uv.x, a_uv.y);
                    b = convert2rgb_TV_range(b_y, b_uv.x, b_uv.y);
                    c = convert2rgb_TV_range(c_y, c_uv.x, c_uv.y);
                    d = convert2rgb_TV_range(d_y, d_uv.x, d_uv.y);
                }

                float3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                  if (norm) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                  } else {
                    norm_tmp.x = out_tmp.z;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.x;
                  }
                } else {
                  if (norm) {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                  } else {
                    norm_tmp.x = out_tmp.x;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.z;
                  }
                }
            }
            t1[v_idx] = norm_tmp.x;
            t2[v_idx] = norm_tmp.y;
            t3[v_idx] = norm_tmp.z;
          }

          if (out_align) {
            int out_idx = h_idx * out_w + tw_idx;
            ((float2*)out)[out_idx] = *((float2*)t1);
            ((float2*)out)[out_idx + out_w * out_h] = *((float2*)t2);
            ((float2*)out)[out_idx + (out_w << 1) * out_h] = *((float2*)t3);
          } else {
            int out_idx = h_idx * out_w + tw_idx;
            out[out_idx] = t1[0];
            out[out_idx + out_w * out_h] = t2[0];
            out[out_idx + (out_w << 1) * out_h] = t3[0];
          }
        }
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, bool out_align = false, bool norm = true>
__global__ void roi_yu122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *u1 = y2 + (roi_w & 0xfffffff8) + 8;
    uint8_t *u2 = u1 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    uint8_t *v1 = u2 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    uint8_t *v2 = v1 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
        int top_h = raw_h_idx;
        int bottom_h = MIN(in_h - 1, top_h + 1);
        float y_diff = raw_h_idx - top_h;
        if (h_run) {
            int half_w = in_w >> 1;
            int uv_h1 = top_h >> 1;
            int uv_h2 = bottom_h >> 1;
          if (align) {
            global2share_copy_align(in + top_h * in_w + roi_w_start, y1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            int offset = in_w * in_h + (roi_w_start >> 1);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
            global2share_copy_align(in + uv_h1 * half_w + offset, u1, copy_size);
            global2share_copy_align(in + uv_h2 * half_w + offset, u2, copy_size);
            offset = in_w * in_h + (in_w * in_h >> 2) + (roi_w_start >> 1);
            global2share_copy_align(in + uv_h1 * half_w + offset, v1, copy_size);
            global2share_copy_align(in + uv_h2 * half_w + offset, v2, copy_size);
          } else {
            int sm_offset = global2share_copy(in + top_h * in_w + roi_w_start, y1, roi_w);
            y1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            y2 += sm_offset;
            int offset = in_w * in_h + (roi_w_start >> 1);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
            sm_offset = global2share_copy(in + uv_h1 * half_w + offset, u1, copy_size);
            u1 += sm_offset;
            sm_offset = global2share_copy(in + uv_h2 * half_w + offset, u2, copy_size);
            u2 += sm_offset;
            offset = in_w * in_h + (in_w * in_h >> 2) + (roi_w_start >> 1);
            sm_offset = global2share_copy(in + uv_h1 *half_w + offset, v1, copy_size);
            v1 += sm_offset;
            sm_offset = global2share_copy(in + uv_h2 * half_w + offset, v2, copy_size);
            v2 += sm_offset;
          }
        }
        __syncthreads();

        int num_v = 1;
        if (out_align) {
          out_w = out_w >> 1;
          num_v = 2;
        }
        for (int tw_idx = threadIdx.x; tw_idx < out_w; tw_idx += blockDim.x) {
          float t1[2], t2[2], t3[2];
          for (int v_idx = 0; v_idx < num_v; v_idx++) {
            int w_idx = tw_idx * num_v + v_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(roi_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                float3 a, b, c, d;
                a.x = y1[left_w];
                b.x = y1[right_w];
                c.x = y2[left_w];
                d.x = y2[right_w];
                int left = (left_w + (roi_w_start & 1))>>1;
                int right = (right_w + (roi_w_start & 1))>>1;

                a.y = u1[left];
                a.z = v1[left];
                b.y = u1[right];
                b.z = v1[right];
                c.y = u2[left];
                c.z = v2[left];
                d.y = u2[right];
                d.z = v2[right];
                if (full_range) {
                    a = convert2rgb_full_range(a.x, a.y, a.z);
                    b = convert2rgb_full_range(b.x, b.y, b.z);
                    c = convert2rgb_full_range(c.x, c.y, c.z);
                    d = convert2rgb_full_range(d.x, d.y, d.z);
                } else {
                    a = convert2rgb_TV_range(a.x, a.y, a.z);
                    b = convert2rgb_TV_range(b.x, b.y, b.z);
                    c = convert2rgb_TV_range(c.x, c.y, c.z);
                    d = convert2rgb_TV_range(d.x, d.y, d.z);
                }
                float3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    if (norm) {
                        norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                        norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                        norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                    } else {
                        norm_tmp.x = out_tmp.z;
                        norm_tmp.y = out_tmp.y;
                        norm_tmp.z = out_tmp.x;
                    }
                } else {
                    if (norm) {
                        norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                        norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                        norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                    } else {
                      norm_tmp.x = out_tmp.x;
                      norm_tmp.y = out_tmp.y;
                      norm_tmp.z = out_tmp.z;
                    }
                }
            }
            t1[v_idx] = norm_tmp.x;
            t2[v_idx] = norm_tmp.y;
            t3[v_idx] = norm_tmp.z;
          }

          if (out_align) {
            int out_idx = h_idx * out_w + tw_idx;
            ((float2*)out)[out_idx] = *((float2*)t1);
            ((float2*)out)[out_idx + out_w * out_h] = *((float2*)t2);
            ((float2*)out)[out_idx + (out_w << 1) * out_h] = *((float2*)t3);
          } else {
            int out_idx = h_idx * out_w + tw_idx;
            out[out_idx] = t1[0];
            out[out_idx + out_w * out_h] = t2[0];
            out[out_idx + (out_w << 1) * out_h] = t3[0];
          }
        }
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, bool out_align = false>
__global__ void roi_yuv444p2rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *u1 = y2 + (roi_w & 0xfffffff8) + 8;
    uint8_t *u2 = u1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *v1 = u2 + (roi_w & 0xfffffff8) + 8;
    uint8_t *v2 = v1 + (roi_w & 0xfffffff8) + 8;
    float *t1 = (float*)(v2 + (roi_w & 0xfffffff8) + 8);
    float *t2 = t1 + out_w;
    float *t3 = t2 + out_w;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
        int top_h = raw_h_idx;
        int bottom_h = MIN(in_h - 1, top_h + 1);
        float y_diff = raw_h_idx - top_h;
        if (h_run) {
          if (align) {
            int offset = roi_w_start;
            global2share_copy_align(in + top_h * in_w + offset, y1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + offset, y2, roi_w);
            offset = in_w * in_h + roi_w_start;
            global2share_copy_align(in + top_h * in_w + offset, u1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + offset, u2, roi_w);
            offset = in_w * in_h * 2 + roi_w_start;
            global2share_copy_align(in + top_h * in_w + offset, v1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + offset, v2, roi_w);
          } else {
            int offset = roi_w_start;
            int sm_offset = global2share_copy(in + top_h * in_w + offset, y1, roi_w);
            y1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + offset, y2, roi_w);
            y2 += sm_offset;
            offset = in_w * in_h + roi_w_start;
            sm_offset = global2share_copy(in + top_h * in_w + offset, u1, roi_w);
            u1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + offset, u2, roi_w);
            u2 += sm_offset;
            offset = in_w * in_h * 2 + roi_w_start;
            sm_offset = global2share_copy(in + top_h * in_w + offset, v1, roi_w);
            v1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + offset, v2, roi_w);
            v2 += sm_offset;
          }
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(roi_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                float3 a, b, c, d;
                a.x = y1[left_w];
                b.x = y1[right_w];
                c.x = y2[left_w];
                d.x = y2[right_w];

                a.y = u1[left_w];
                a.z = v1[left_w];
                b.y = u1[right_w];
                b.z = v1[right_w];
                c.y = u2[left_w];
                c.z = v2[left_w];
                d.y = u2[right_w];
                d.z = v2[right_w];
                if (full_range) {
                    a = convert2rgb_full_range(a.x, a.y, a.z);
                    b = convert2rgb_full_range(b.x, b.y, b.z);
                    c = convert2rgb_full_range(c.x, c.y, c.z);
                    d = convert2rgb_full_range(d.x, d.y, d.z);
                } else {
                    a = convert2rgb_TV_range(a.x, a.y, a.z);
                    b = convert2rgb_TV_range(b.x, b.y, b.z);
                    c = convert2rgb_TV_range(c.x, c.y, c.z);
                    d = convert2rgb_TV_range(d.x, d.y, d.z);
                }
                float3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            if (out_align) {
                t1[w_idx] = norm_tmp.x;
                t2[w_idx] = norm_tmp.y;
                t3[w_idx] = norm_tmp.z;
            } else { 
                int out_idx = h_idx * out_w + w_idx;
                out[out_idx] = norm_tmp.x;
                out[out_idx + out_w * out_h] = norm_tmp.y;
                out[out_idx + out_w * out_h * 2] = norm_tmp.z;
            }
        }
        __syncthreads();
        if (out_align) {
            for (int w_idx = threadIdx.x; w_idx < (out_w>>1); w_idx += blockDim.x) {
                int out_idx = h_idx * (out_w>>1) + w_idx;
                ((float2*)out)[out_idx] = ((float2*)t1)[w_idx];
                ((float2*)out)[out_idx + (out_w * out_h>>1)] = ((float2*)t2)[w_idx];
                ((float2*)out)[out_idx + out_w * out_h] = ((float2*)t3)[w_idx];
            }
        }
    }
}

template <bool align = false, bool out_align = false>
__global__ void roi_yuv400p2rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean, float std, float scale,
    float y_ratio,float x_ratio, float pad) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (roi_w & 0xfffffff8) + 8;
    float *t1 = (float*)(y2 + (roi_w & 0xfffffff8) + 8);
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
        int top_h = raw_h_idx;
        int bottom_h = MIN(in_h - 1, top_h + 1);
        float y_diff = raw_h_idx - top_h;
        if (h_run) {
          if (align) {
            global2share_copy_align(in + top_h * in_w + roi_w_start, y1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + roi_w_start, y2, roi_w);
          } else {
            int sm_offset = global2share_copy(in + top_h * in_w + roi_w_start, y1, roi_w);
            y1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            y2 += sm_offset;
          }
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float norm_tmp = pad;
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(roi_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                float a, b, c, d;
                a = y1[left_w];
                b = y1[right_w];
                c = y2[left_w];
                d = y2[right_w];
                float out_tmp;
                out_tmp = a * scale1 + b * scale2 + c * scale3 + d * scale4;
                norm_tmp = (out_tmp * scale - mean) * std;
            }
            if (out_align) {
                t1[w_idx] = norm_tmp;
            } else { 
                int out_idx = h_idx * out_w + w_idx;
                out[out_idx] = norm_tmp;
            }
        }
        __syncthreads();
        if (out_align) {
            for (int w_idx = threadIdx.x; w_idx < (out_w>>1); w_idx += blockDim.x) {
                int out_idx = h_idx * (out_w>>1) + w_idx;
                ((float2*)out)[out_idx] = ((float2*)t1)[w_idx];
            }
        }
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, bool out_align = false>
__global__ void roi_yuv422p2rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *u1 = y2 + (roi_w & 0xfffffff8) + 8;
    uint8_t *u2 = u1 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    uint8_t *v1 = u2 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    uint8_t *v2 = v1 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    float *t1 = (float*)(v2 + ((roi_w + 1) / 2 & 0xfffffff8) + 8);
    float *t2 = t1 + out_w;
    float *t3 = t2 + out_w;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
        int top_h = raw_h_idx;
        int bottom_h = MIN(in_h - 1, top_h + 1);
        float y_diff = raw_h_idx - top_h;
        if (h_run) {
            int half_w = in_w >> 1;
            int uv_h1 = top_h;
            int uv_h2 = bottom_h;
          if (align) {
            global2share_copy_align(in + top_h * in_w + roi_w_start, y1, roi_w);
            global2share_copy_align(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            int offset = in_w * in_h + (roi_w_start >> 1);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
            global2share_copy_align(in + uv_h1 * half_w + offset, u1, copy_size);
            global2share_copy_align(in + uv_h2 * half_w + offset, u2, copy_size);
            offset = in_w * in_h + (in_w * in_h >> 1) + (roi_w_start >> 1);
            global2share_copy_align(in + uv_h1 * half_w + offset, v1, copy_size);
            global2share_copy_align(in + uv_h2 * half_w + offset, v2, copy_size);
          } else {
            int sm_offset = global2share_copy(in + top_h * in_w + roi_w_start, y1, roi_w);
            y1 += sm_offset;
            sm_offset = global2share_copy(in + bottom_h * in_w + roi_w_start, y2, roi_w);
            y2 += sm_offset;
            int offset = in_w * in_h + (roi_w_start >> 1);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
            sm_offset = global2share_copy(in + uv_h1 * half_w + offset, u1, copy_size);
            u1 += sm_offset;
            sm_offset = global2share_copy(in + uv_h2 * half_w + offset, u2, copy_size);
            u2 += sm_offset;
            offset = in_w * in_h + (in_w * in_h >> 1) + (roi_w_start >> 1);
            sm_offset = global2share_copy(in + uv_h1 *half_w + offset, v1, copy_size);
            v1 += sm_offset;
            sm_offset = global2share_copy(in + uv_h2 * half_w + offset, v2, copy_size);
            v2 += sm_offset;
          }
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(roi_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                float3 a, b, c, d;
                a.x = y1[left_w];
                b.x = y1[right_w];
                c.x = y2[left_w];
                d.x = y2[right_w];
                int left = (left_w + (roi_w_start & 1))>>1;
                int right = (right_w + (roi_w_start & 1))>>1;

                a.y = u1[left];
                a.z = v1[left];
                b.y = u1[right];
                b.z = v1[right];
                c.y = u2[left];
                c.z = v2[left];
                d.y = u2[right];
                d.z = v2[right];
                if (full_range) {
                    a = convert2rgb_full_range(a.x, a.y, a.z);
                    b = convert2rgb_full_range(b.x, b.y, b.z);
                    c = convert2rgb_full_range(c.x, c.y, c.z);
                    d = convert2rgb_full_range(d.x, d.y, d.z);
                } else {
                    a = convert2rgb_TV_range(a.x, a.y, a.z);
                    b = convert2rgb_TV_range(b.x, b.y, b.z);
                    c = convert2rgb_TV_range(c.x, c.y, c.z);
                    d = convert2rgb_TV_range(d.x, d.y, d.z);
                }
                float3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            if (out_align) {
                t1[w_idx] = norm_tmp.x;
                t2[w_idx] = norm_tmp.y;
                t3[w_idx] = norm_tmp.z;
            } else { 
                int out_idx = h_idx * out_w + w_idx;
                out[out_idx] = norm_tmp.x;
                out[out_idx + out_w * out_h] = norm_tmp.y;
                out[out_idx + out_w * out_h * 2] = norm_tmp.z;
            }
        }
        __syncthreads();
        if (out_align) {
            for (int w_idx = threadIdx.x; w_idx < (out_w>>1); w_idx += blockDim.x) {
                int out_idx = h_idx * (out_w>>1) + w_idx;
                ((float2*)out)[out_idx] = ((float2*)t1)[w_idx];
                ((float2*)out)[out_idx + (out_w * out_h>>1)] = ((float2*)t2)[w_idx];
                ((float2*)out)[out_idx + out_w * out_h] = ((float2*)t3)[w_idx];
            }
        }
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, bool out_align = false>
__global__ void roi_yuv4222rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int roi_w_pad_left = roi_w_start & 1;
    int roi_w_pad_right = (roi_w_start + roi_w) & 1;
    int copy_size = (roi_w_pad_left + roi_w + roi_w_pad_right) * 2;

    extern __shared__ uint8_t sm[];
    uint8_t *line1 = sm;
    uint8_t *line2 = line1 + (copy_size & 0xfffffff8) + 8;
    float *t1 = (float*)(line2 + (copy_size & 0xfffffff8) + 8);
    float *t2 = t1 + out_w;
    float *t3 = t2 + out_w;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    if (h_idx < out_h) {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
        int top_h = raw_h_idx;
        int bottom_h = MIN(in_h - 1, top_h + 1);
        float y_diff = raw_h_idx - top_h;
        if (h_run) {
          if (align) {
            global2share_copy_align(in + (top_h * in_w + roi_w_start - roi_w_pad_left) * 2, line1, copy_size);
            global2share_copy_align(in + (bottom_h * in_w + roi_w_start - roi_w_pad_left) * 2, line2, copy_size);
          } else {
            int sm_offset = global2share_copy(in + (top_h * in_w + roi_w_start - roi_w_pad_left) * 2, line1, copy_size);
            line1 += sm_offset;
            sm_offset = global2share_copy(in + (bottom_h * in_w + roi_w_start - roi_w_pad_left) * 2, line2, copy_size);
            line2 += sm_offset;
          }
        }
        __syncthreads();
        for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
            int out_idx = h_idx * out_w + w_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(roi_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                float3 a, b, c, d;
                int offset = (left_w + roi_w_pad_left) << 1;
                a.x = line1[offset];
                c.x = line2[offset];
                offset = (right_w + roi_w_pad_left) << 1;
                b.x = line1[offset];
                d.x = line2[offset];
                offset = ((left_w + roi_w_pad_left) & 0xfffffffe)*2 + 1;
                a.y = line1[offset];
                a.z = line1[offset + 2];
                c.y = line2[offset];
                c.z = line2[offset + 2];
                offset = ((right_w + roi_w_pad_left) & 0xfffffffe)*2 + 1;
                b.y = line1[offset];
                b.z = line1[offset + 2];
                d.y = line2[offset];
                d.z = line2[offset + 2];
                if (full_range) {
                    a = convert2rgb_full_range(a.x, a.y, a.z);
                    b = convert2rgb_full_range(b.x, b.y, b.z);
                    c = convert2rgb_full_range(c.x, c.y, c.z);
                    d = convert2rgb_full_range(d.x, d.y, d.z);
                } else {
                    a = convert2rgb_TV_range(a.x, a.y, a.z);
                    b = convert2rgb_TV_range(b.x, b.y, b.z);
                    c = convert2rgb_TV_range(c.x, c.y, c.z);
                    d = convert2rgb_TV_range(d.x, d.y, d.z);
                }
                float3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                }
            }
            if (out_align) {
                t1[w_idx] = norm_tmp.x;
                t2[w_idx] = norm_tmp.y;
                t3[w_idx] = norm_tmp.z;
            } else { 
                int out_idx = h_idx * out_w + w_idx;
                out[out_idx] = norm_tmp.x;
                out[out_idx + out_w * out_h] = norm_tmp.y;
                out[out_idx + out_w * out_h * 2] = norm_tmp.z;
            }
        }
        __syncthreads();
        if (out_align) {
            for (int w_idx = threadIdx.x; w_idx < (out_w>>1); w_idx += blockDim.x) {
                int out_idx = h_idx * (out_w>>1) + w_idx;
                ((float2*)out)[out_idx] = ((float2*)t1)[w_idx];
                ((float2*)out)[out_idx + (out_w * out_h>>1)] = ((float2*)t2)[w_idx];
                ((float2*)out)[out_idx + out_w * out_h] = ((float2*)t3)[w_idx];
            }
        }
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, int out_align = ALIGN1B, bool norm = true>
__global__ void roi_nv122rgba_kernel(uint8_t* __restrict__ in, uchar4* __restrict__ out,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *uv1 = y1 + (roi_w & 0xfffffff8) + 8;
    int h_idx = blockIdx.x; 
    if (align) {
        global2share_copy_align(in + (h_idx + roi_h_start) * in_w + roi_w_start, y1, roi_w);
        int copy_size = (roi_w + 1 + (roi_w_start & 1)) & 0xfffffffe;
        global2share_copy_align(in + in_w * in_h + ((h_idx + roi_h_start) >> 1) * in_w + (roi_w_start & 0xfffffffe), uv1, copy_size);
    } else {
        int sm_offset = global2share_copy(in + (h_idx + roi_h_start) * in_w + roi_w_start, y1, roi_w);
        y1 += sm_offset;
        int copy_size = (roi_w + 1 + (roi_w_start & 1)) & 0xfffffffe;
        sm_offset = global2share_copy(in + in_w * in_h + ((h_idx + roi_h_start) >> 1) * in_w + (roi_w_start & 0xfffffffe), uv1, copy_size);
        uv1 += sm_offset;
    }
    __syncthreads();

    for (int w_idx = threadIdx.x; w_idx < roi_w; w_idx += blockDim.x) {
        int uv_w_idx = (w_idx + (roi_w_start & 1))>>1;
        uchar2 a_uv = ((uchar2*)uv1)[uv_w_idx];
        float3 a;

        float a_y = y1[w_idx];
        if (full_range) {
            a = convert2rgb_full_range(a_y, a_uv.x, a_uv.y);
        } else {
            a = convert2rgb_TV_range(a_y, a_uv.x, a_uv.y);
        }
        uchar4 out_tmp;
        out_tmp.x = a.x;
        out_tmp.y = a.y;
        out_tmp.z = a.z;
        int out_idx = h_idx * roi_w + w_idx;
        out[out_idx] = out_tmp;
    }
}


template <bool bgr_format = false, bool full_range = false, bool align = false>
__global__ void roi_yu122rgba_kernel(uint8_t* __restrict__ in, uchar4* __restrict__ out, 
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *u1 = y1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *v1 = u1 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    int h_idx = blockIdx.x;
    int half_w = in_w >> 1;
    int uv_h1 = (h_idx + roi_h_start) >> 1;
    if (align) {
        global2share_copy_align(in + (h_idx + roi_h_start) * in_w + roi_w_start, y1, roi_w);
        int offset = in_w * in_h + (roi_w_start >> 1);
        int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
        global2share_copy_align(in + uv_h1 * half_w + offset, u1, copy_size);
        offset = in_w * in_h + (in_w * in_h >> 2) + (roi_w_start >> 1);
        global2share_copy_align(in + uv_h1 * half_w + offset, v1, copy_size);
    } else {
        int sm_offset = global2share_copy(in + (h_idx + roi_h_start) * in_w + roi_w_start, y1, roi_w);
        y1 += sm_offset;
        int offset = in_w * in_h + (roi_w_start >> 1);
        int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
        sm_offset = global2share_copy(in + uv_h1 * half_w + offset, u1, copy_size);
        u1 += sm_offset;
        offset = in_w * in_h + (in_w * in_h >> 2) + (roi_w_start >> 1);
        sm_offset = global2share_copy(in + uv_h1 *half_w + offset, v1, copy_size);
        v1 += sm_offset;
    }
    __syncthreads();
    for (int w_idx = threadIdx.x; w_idx < roi_w; w_idx += blockDim.x) {
        float3 a;
        a.x = y1[w_idx];
        int uv_w = (w_idx + (roi_w_start & 1))>>1;
        a.y = u1[uv_w];
        a.z = v1[uv_w];
        if (full_range) {
            a = convert2rgb_full_range(a.x, a.y, a.z);
        } else {
            a = convert2rgb_TV_range(a.x, a.y, a.z);
        }
        uchar4 out_tmp;
        out_tmp.x = a.x;
        out_tmp.y = a.y;
        out_tmp.z = a.z;
        int out_idx = h_idx * roi_w + w_idx;
        out[out_idx] = out_tmp;
    }
}

template <bool bgr_format = false, bool full_range = false, bool align = false, int out_align = ALIGN1B, bool norm = true>
__global__ void rgba_resize_norm_quantize_fuse_kernel(uchar4* __restrict__ in, uint8_t* __restrict__ out,
        int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float y_ratio,float x_ratio, float pad1, float pad2, float pad3, float zero_point, float scales_input) {
    extern __shared__ uint8_t sm[];
    uchar4 * sm_line = (uchar4*)sm;
    int h_idx = blockIdx.x;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
    int top_h = raw_h_idx;
    int bottom_h = MIN(in_h - 1, top_h + 1);
    float y_diff = raw_h_idx - top_h;
    int line_id = 0;
    if (h_run) {
        int copy_size = in_w * 4;
        if (top_h + 1 < in_h) {
            copy_size *= 2;
            line_id = 1;
        } 
        if (align) {
            global2share_copy_align((uint8_t*)(in + top_h * in_w), sm, copy_size);
        } else {
            int offset = global2share_copy((uint8_t*)(in + top_h * in_w), sm, copy_size);
            sm_line = (uchar4*)(sm + offset);
        }
    }
    __syncthreads();

    const int num_v = 1 << out_align;
    out_w = out_w >> out_align;
    for (int tw_idx = threadIdx.x; tw_idx < out_w; tw_idx += blockDim.x) {
        uint8_t t1[num_v], t2[num_v], t3[num_v];
        for (int v_idx = 0; v_idx < num_v; v_idx++) {
            int w_idx = tw_idx * num_v + v_idx;
            float3 norm_tmp = {pad1, pad2, pad3};
            bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
                int left_w = raw_w_idx;
                int right_w = MIN(in_w - 1, left_w + 1);
                float x_diff = raw_w_idx - left_w;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                uchar4 a, b, c, d;
                a = sm_line[left_w];
                b = sm_line[right_w];
                c = sm_line[left_w +  line_id * in_w];
                d = sm_line[right_w + line_id * in_w];

                float3 out_tmp;
                out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
                out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
                out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
                if (bgr_format) {
                    if (norm) {
                        norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                        norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                        norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                    } else {
                        norm_tmp.x = out_tmp.z;
                        norm_tmp.y = out_tmp.y;
                        norm_tmp.z = out_tmp.x;
                    }
                } else {
                    if (norm) {
                        norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                        norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                        norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                    } else {
                        norm_tmp.x = out_tmp.x;
                        norm_tmp.y = out_tmp.y;
                        norm_tmp.z = out_tmp.z;
                    }
                }
            }
            t1[v_idx] = norm_tmp.x * scales_input + zero_point;
            t2[v_idx] = norm_tmp.y * scales_input + zero_point;
            t3[v_idx] = norm_tmp.z * scales_input + zero_point;
        }
        int out_idx = h_idx * out_w + tw_idx;
        if (out_align == ALIGN1B) {
            out[out_idx] = t1[0];
            out[out_idx + out_w * out_h] = t2[0];
            out[out_idx + (out_w << 1) * out_h] = t3[0];
        } else if (out_align == ALIGN2B){
            ((uint16_t*)out)[out_idx] = *((uint16_t*)t1);
            ((uint16_t*)out)[out_idx + out_w * out_h] = *((uint16_t*)t2);
            ((uint16_t*)out)[out_idx + (out_w << 1) * out_h] = *((uint16_t*)t3);
        } else if (out_align == ALIGN4B){
            ((uint32_t*)out)[out_idx] = *((uint32_t*)t1);
            ((uint32_t*)out)[out_idx + out_w * out_h] = *((uint32_t*)t2);
            ((uint32_t*)out)[out_idx + (out_w << 1) * out_h] = *((uint32_t*)t3);
        } else if (out_align == ALIGN8B){
            ((uint64_t*)out)[out_idx] = *((uint64_t*)t1);
            ((uint64_t*)out)[out_idx + out_w * out_h] = *((uint64_t*)t2);
            ((uint64_t*)out)[out_idx + (out_w << 1) * out_h] = *((uint64_t*)t3);
        }
    }
}

#define PRECISION_BITS (32 - 8 - 2)

__device__ __forceinline__ uint8_t
clip8(int in) {
    return in >> PRECISION_BITS;
}

__device__ __forceinline__ float bilinear_filter(float x) {
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return 1.0f - x;
    }
    return 0.0f;
}


template <int A>
struct Int2Type
{
   enum {VALUE = A};
};

template <int KSIZE>               ///< The PTX compute capability for which to to specialize this collective
struct KernelCoeff
{
private:
	enum {
		STEPS = KSIZE,
	};

	template <int STEP>
	__device__ __forceinline__ void compute_coeffs_step(
			float* k, float* ww, int xmin, float center, float ss, int in_size, Int2Type<STEP>)
	{
		k[STEP] = 0.0f;
		if (xmin + STEP < in_size) {
			float w = bilinear_filter((STEP + xmin - center + 0.5f) * ss);
			k[STEP] = w;
			*ww += w;
		}

		compute_coeffs_step(k, ww, xmin, center, ss, in_size, Int2Type<STEP + 1>());
	}

	__device__ __forceinline__ void compute_coeffs_step(
			float* k, float* ww, int xmin, float center, float ss, int in_size, Int2Type<STEPS>)
	{
	   return;
	}

	template <int STEP>
	__device__ __forceinline__ void norm_coeffs_step(
			float* k, float ww, int xmin, int in_size, Int2Type<STEP>)
	{
		if (xmin + STEP < in_size) {
			k[STEP] *= ww;
		}

		norm_coeffs_step(k, ww, xmin, in_size, Int2Type<STEP + 1>());
	}

	__device__ __forceinline__ void norm_coeffs_step(
			float* k, float ww, int xmin, int in_size, Int2Type<STEPS>)
	{
	   return;
	}

public:
    __device__ __forceinline__ KernelCoeff()
    {}

	template <int STEP>
	__device__ __forceinline__ void horiz_mad_coeffs_step(
			int3* ss, uchar4* data, float* k, int xmin, int in_size, Int2Type<STEP>)
	{
		if (xmin + STEP < in_size) {
			int coeff = (int)((k[STEP] < 0.f ? -0.5f : 0.5f)  + k[STEP] * (1 << PRECISION_BITS));
			ss->x += data[xmin + STEP].x * coeff;
			ss->y += data[xmin + STEP].y * coeff;
			ss->z += data[xmin + STEP].z * coeff;
		}

		horiz_mad_coeffs_step(ss, data, k, xmin, in_size, Int2Type<STEP + 1>());
	}

	__device__ __forceinline__ void horiz_mad_coeffs_step(
			int3* ss, uchar4* data, float* k, int xmin, int in_size, Int2Type<STEPS>)
	{
	   return;
	}

	template <int STEP>
	__device__ __forceinline__ void vert_mad_coeffs_step(
			int3* ss, uchar4* data, float* k, int offset, int ymin, int in_size, int stride, Int2Type<STEP>)
	{
		if (ymin + STEP < in_size) {
			int coeff = (int)((k[STEP] < 0.f ? -0.5f : 0.5f)  + k[STEP] * (1 << PRECISION_BITS));
            uchar4 t = data[(ymin + STEP) * stride + offset];
			ss->x += t.x * coeff;
			ss->y += t.y * coeff;
			ss->z += t.z * coeff;
		}

		vert_mad_coeffs_step(ss, data, k, offset, ymin, in_size, stride, Int2Type<STEP + 1>());
	}

	__device__ __forceinline__ void vert_mad_coeffs_step(
			int3* ss, uchar4* data, float* k, int offset, int ymin, int in_size, int stride, Int2Type<STEPS>)
	{
	   return;
	}

	__device__ __forceinline__ void compute_coeffs(
			int xx, int in_size, float scale,
			float* k, int *xmin) {
		float center = (xx + 0.5f) * scale;
		float ww = 0.0f;
		// Round the value
		*xmin = (int)(center - scale + 0.5f);
		if (*xmin < 0) {
			*xmin = 0;
		}

		compute_coeffs_step(k, &ww, *xmin, center, 1.0f / scale, in_size, Int2Type<0>());
		norm_coeffs_step(k, 1.f / ww, *xmin, in_size, Int2Type<0>());
	}
};


template <bool bgr_format = false, bool full_range = false, bool align = false>
__global__ void roi_yu122rgba_pad_kernel(uint8_t* __restrict__ in, uchar4* __restrict__ out, 
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h, 
    int pad_w, int pad_h, uint8_t pad1, uint8_t pad2, uint8_t pad3) {
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *u1 = y1 + (roi_w & 0xfffffff8) + 8;
    uint8_t *v1 = u1 + ((roi_w + 1) / 2 & 0xfffffff8) + 8;
    int h_idx = (int)blockIdx.x - pad_h;
    bool h_run = h_idx >= 0 && h_idx < roi_h;
    if (h_run) {
        int half_w = in_w >> 1;
        int uv_h1 = (h_idx + roi_h_start) >> 1;
        if (align) {
            global2share_copy_align(in + (h_idx + roi_h_start) * in_w + roi_w_start, y1, roi_w);
            int offset = in_w * in_h + (roi_w_start >> 1);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
            global2share_copy_align(in + uv_h1 * half_w + offset, u1, copy_size);
            offset = in_w * in_h + (in_w * in_h >> 2) + (roi_w_start >> 1);
            global2share_copy_align(in + uv_h1 * half_w + offset, v1, copy_size);
        } else {
            int sm_offset = global2share_copy(in + (h_idx + roi_h_start) * in_w + roi_w_start, y1, roi_w);
            y1 += sm_offset;
            int offset = in_w * in_h + (roi_w_start >> 1);
            int copy_size = (roi_w + 1 + (roi_w_start & 1)) >> 1;
            sm_offset = global2share_copy(in + uv_h1 * half_w + offset, u1, copy_size);
            u1 += sm_offset;
            offset = in_w * in_h + (in_w * in_h >> 2) + (roi_w_start >> 1);
            sm_offset = global2share_copy(in + uv_h1 *half_w + offset, v1, copy_size);
            v1 += sm_offset;
        }
    }
    __syncthreads();
    for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
        uchar4 out_tmp = {pad1, pad2, pad3, 0};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + roi_w);
        if (h_run && w_run) {
            float3 a;
            a.x = y1[w_idx - pad_w];
            int uv_w = (w_idx - pad_w + (roi_w_start & 1))>>1;
            a.y = u1[uv_w];
            a.z = v1[uv_w];
            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
            }
            out_tmp.x = a.x;
            out_tmp.y = a.y;
            out_tmp.z = a.z;
        }
        int out_idx = h_idx * out_w + w_idx;
        out[out_idx] = out_tmp;
    }
}


template <int KSIZE = 1>
__global__ void rgba_horizontal_resize_kernel(uchar4* __restrict__ in, uchar4* __restrict__ out,
        int in_w, int in_h, int out_w, int out_h, float filter_scale) {
    typedef KernelCoeff<KSIZE> KernelCoeffT;
    KernelCoeffT kernel_coeff = KernelCoeffT();
    int horiz_bounds[1];
	float horiz_coeffs[KSIZE];

	int h_idx = blockIdx.y;
    for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
        kernel_coeff.compute_coeffs(w_idx, in_w, filter_scale, horiz_coeffs, horiz_bounds);
        if (w_idx < out_w) {
            int xmin = horiz_bounds[0];
            int3 ss = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1)};
            kernel_coeff.horiz_mad_coeffs_step(&ss, in + h_idx * in_w, horiz_coeffs, horiz_bounds[0], in_w, Int2Type<0>());

            int out_idx = h_idx * out_w + w_idx;
            out[out_idx] = {clip8(ss.x), clip8(ss.y), clip8(ss.z), 0};
        }
    }
}


template <bool bgr_format = false, bool full_range = false, bool norm = true, int KSIZE=1>
__global__ void rgba_vertical_resize_kernel(uchar4* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int out_w, int out_h, float filter_scale,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale) {
    typedef KernelCoeff<KSIZE> KernelCoeffT;
    KernelCoeffT kernel_coeff = KernelCoeffT();
	float coeffs[KSIZE];
	int bounds[1];

    int h_idx = blockIdx.y;
    for (int tw_idx = threadIdx.x; tw_idx < out_w; tw_idx += blockDim.x) {
        kernel_coeff.compute_coeffs(h_idx, in_h, filter_scale, coeffs, bounds);
        int3 ss = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1)};
        kernel_coeff.vert_mad_coeffs_step(&ss, in, coeffs, tw_idx, bounds[0], in_h, in_w, Int2Type<0>());

        float3 out_tmp;
        out_tmp.x = clip8(ss.x);
        out_tmp.y = clip8(ss.y);
        out_tmp.z = clip8(ss.z);

        float3 norm_tmp = {0, 0, 0};
        if (bgr_format) {
            if (norm) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = out_tmp.z;
                norm_tmp.y = out_tmp.y;
                norm_tmp.z = out_tmp.x;
            }
        } else {
            if (norm) {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            } else {
                norm_tmp.x = out_tmp.x;
                norm_tmp.y = out_tmp.y;
                norm_tmp.z = out_tmp.z;
            }
        }

        int out_idx = h_idx * out_w + tw_idx;
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + (out_w << 1) * out_h] = norm_tmp.z;
    }
}
#endif

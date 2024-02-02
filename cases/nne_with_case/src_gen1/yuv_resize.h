#ifndef YUV_RESIZE_H_
#define YUV_RESIZE_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"

template<bool align = true>
__global__  void resize_kernel_nv12_nv21(const uint8_t *__restrict__ input,
        int h,
        int w,
        int t_h,
        int t_w,
        float t_h_trans,
        float t_w_trans,
        uint8_t *__restrict__ out) {

    const uint8_t * __restrict__ in_Y = input;
    const uint8_t * __restrict__ in_UV = input + h * w;
    uint8_t * __restrict__ out_Y = out;
    uint16_t * __restrict__ out_UV = (uint16_t*)(out + t_h * t_w);
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (w & 0xfffffff8) + 16;
    uint8_t *uv = y2 + (w & 0xfffffff8) + 16;
    int h_idx = blockIdx.x;

    int dst_h_idx = h_idx * 2;
    int src_h_idx1 = dst_h_idx * t_h_trans;
    int src_h_idx2  = (dst_h_idx + 1) * t_h_trans;
    int uv_src_h_idx = src_h_idx1 >> 1;
    if (align) {
        global2share_copy_align(in_Y + src_h_idx1 * w , y1, w);
        global2share_copy_align(in_Y + src_h_idx2 * w , y2, w);
        global2share_copy_align(in_UV + w * (src_h_idx1 >> 1), uv, w);
    } else {
        int offset = global2share_copy(in_Y + src_h_idx1 * w , y1, w);
        y1 += offset;
        offset = global2share_copy(in_Y + src_h_idx2 * w , y2, w);
        y2 += offset;
        offset = global2share_copy(in_UV + w * (src_h_idx1 >> 1), uv, w);
        uv += offset;
    }
    __syncthreads();
    for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
        int src_w_idx = w_idx * t_w_trans;
        int out_offset = dst_h_idx * t_w + w_idx;

        out_Y[out_offset] = y1[src_w_idx];      
        out_Y[out_offset + t_w] = y2[src_w_idx];

        int uv_in_offset = src_w_idx >> 1;
        int uv_offset = h_idx * (t_w >> 1) + (w_idx>>1);
        if((w_idx & 0x1) == 0)
        {
            out_UV[uv_offset] = ((uint16_t*)uv)[uv_in_offset];
        }
    }
}

template<bool align = true>
__global__  void resize_kernel_i420(const uint8_t *__restrict__ input,
                                        int h,
                                        int w,
                                        int t_h,
                                        int t_w,
                                        float t_h_trans,
                                        float t_w_trans,
                                        uint8_t *__restrict__ out) {
    const uint8_t * __restrict__ in_Y = input;
    const uint8_t * __restrict__ in_U = in_Y + h * w;
    const uint8_t * __restrict__ in_V = in_U + (h * w >> 2);
    uint8_t * __restrict__ out_Y = out;
    uint8_t * __restrict__ out_U = out + t_h * t_w;
    uint8_t * __restrict__ out_V = out_U + (t_h * t_w >> 2);
    extern __shared__ uint8_t sm[];
    uint8_t *y1 = sm;
    uint8_t *y2 = y1 + (w & 0xfffffff8) + 16;
    uint8_t *u = y2 + (w & 0xfffffff8) + 16;
    uint8_t *v = u + (w & 0xfffffff8) + 16;
    int h_idx = blockIdx.x;
    int dst_h_idx = h_idx * 2;
    int src_h_idx1 = dst_h_idx * t_h_trans;
    int src_h_idx2  = (dst_h_idx + 1) * t_h_trans;
    int uv_src_h_idx = src_h_idx1 >> 1;
    if (align) {
        global2share_copy_align(in_Y + src_h_idx1 * w , y1, w);
        global2share_copy_align(in_Y + src_h_idx2 * w , y2, w);
        global2share_copy_align(in_U + (w >> 1) * (src_h_idx1 >> 1), u, (w >> 1));
        global2share_copy_align(in_V + (w >> 1) * (src_h_idx1 >> 1), v, (w >> 1));
    } else {
        int offset = global2share_copy(in_Y + src_h_idx1 * w , y1, w);
        y1 += offset;
        offset = global2share_copy(in_Y + src_h_idx2 * w , y2, w);
        y2 += offset;
        offset = global2share_copy(in_U + (w >> 1) * (src_h_idx1 >> 1), u, (w >> 1));
        u += offset;
        offset = global2share_copy(in_V + (w >> 1) * (src_h_idx1 >> 1), v, (w >> 1));
        v += offset;
    }
    __syncthreads();
    for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
        int src_w_idx = w_idx * t_w_trans;
        int out_offset = dst_h_idx * t_w + w_idx;

        out_Y[out_offset] = y1[src_w_idx];      //[0,0]
        out_Y[out_offset + t_w] = y2[src_w_idx];//[1,0]

        int uv_in_offset = src_w_idx >> 1;
        int uv_offset = h_idx * (t_w >> 1) + (w_idx>>1);
        if((w_idx & 0x1) == 0)
        {
            out_U[uv_offset] = u[uv_in_offset];
            out_V[uv_offset] = v[uv_in_offset];
        }
    }//end of w
}

template<bool align = true>
__global__  void bilinear_resize_kernel_nv12_nv21(const uint8_t *__restrict__ input,
        int h,
        int w,
        int t_h,
        int t_w,
        float t_h_trans,
        float t_w_trans,
        uint8_t *__restrict__ out) {

    const uint8_t * __restrict__ in_Y = input;
    const uint8_t * __restrict__ in_UV = input + h * w;
    uint8_t * __restrict__ out_Y = out;
    uchar2 * __restrict__ out_UV = (uchar2*)(out + t_h * t_w);
    extern __shared__ uint8_t sm[];
    uint8_t *y = sm;
    uint8_t *uv = y + (w * 2 & 0xfffffff8) + 16;
    int h_idx = blockIdx.x;

    float src_h_idx = h_idx * t_h_trans;
    int top_h = src_h_idx;
    int uv_src_h_idx = top_h >> 1;
    int line_id_y = 0;
    int line_id_uv = 1;
    int copy_size = w;
    if (top_h + 1 < h) {
        copy_size *= 2;
        line_id_y = 1;
    }

    int offset = 0;
    if (align) {
        global2share_copy_align(in_Y + top_h * w , y, copy_size);
    } else {
        offset = global2share_copy(in_Y + top_h * w , y, copy_size);
        y += offset;
    }
    if ((h_idx & 1) == 0) {
        copy_size = w * 2;
        if ((top_h & 1) == 0 || top_h + 1 == h ) {
            line_id_uv = 0;
            copy_size = w;
        }
        if (align) {
            global2share_copy_align(in_UV + w * (top_h >> 1), uv, copy_size);
        } else {
            offset = global2share_copy(in_UV + w * (top_h >> 1), uv, copy_size);
            uv += offset;
        }
    }
    __syncthreads();
    for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
        float src_w_idx = w_idx * t_w_trans;
        int left_w   = src_w_idx;
        int right_w  = MIN(w - 1, left_w + 1);
        int out_offset = h_idx * t_w + w_idx;
        half2 x1, x2;
        x1.x = y[left_w];
        x2.x = y[right_w];
        x1.y = y[left_w  + w * line_id_y];
        x2.y = y[right_w + w * line_id_y];

        half2 w_bi, alpha_h;
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left_w);
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out[out_offset] = __half2int_rn(
                (w_bi.y - w_bi.x) * __float2half_rn(src_h_idx - top_h) + w_bi.x);


        if((w_idx & 1) == 0 && (h_idx & 1) == 0) {
            uchar2 a_uv, b_uv, c_uv, d_uv, out_tmp;
            a_uv = ((uchar2*)uv)[left_w>>1];
            b_uv = ((uchar2*)uv)[right_w>>1];
            c_uv = ((uchar2*)uv)[line_id_uv * (w>>1) + (left_w>>1)];
            d_uv = ((uchar2*)uv)[line_id_uv * (w>>1) + (right_w>>1)];
            x1.x = a_uv.x;
            x2.x = b_uv.x;
            x1.y = c_uv.x;
            x2.y = d_uv.x;
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_tmp.x = __half2int_rn(
                (w_bi.y - w_bi.x) * __float2half_rn(src_h_idx - top_h) + w_bi.x);
            x1.x = a_uv.y;
            x2.x = b_uv.y;
            x1.y = c_uv.y;
            x2.y = d_uv.y;
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_tmp.y = __half2int_rn(
                (w_bi.y - w_bi.x) * __float2half_rn(src_h_idx - top_h) + w_bi.x);
            int uv_offset = (h_idx >> 1) * (t_w >> 1) + (w_idx>>1);
            out_UV[uv_offset] = out_tmp;
        }
    }
}

template<bool align = true>
__global__  void bilinear_resize_kernel_i420(const uint8_t *__restrict__ input,
                                        int h,
                                        int w,
                                        int t_h,
                                        int t_w,
                                        float t_h_trans,
                                        float t_w_trans,
                                        uint8_t *__restrict__ out) {
    const uint8_t * __restrict__ in_Y = input;
    const uint8_t * __restrict__ in_U = in_Y + h * w;
    const uint8_t * __restrict__ in_V = in_U + (h * w >> 2);
    uint8_t * __restrict__ out_Y = out;
    uint8_t * __restrict__ out_U = out + t_h * t_w;
    uint8_t * __restrict__ out_V = out_U + (t_h * t_w >> 2);
    extern __shared__ uint8_t sm[];
    uint8_t *y = sm;
    uint8_t *u = y + (w * 2 & 0xfffffff8) + 16;
    uint8_t *v = u + (w & 0xfffffff8) + 16;
    int h_idx = blockIdx.x;
    float src_h_idx = h_idx * t_h_trans;
    int top_h = src_h_idx;
    int uv_src_h_idx = top_h >> 1;
    int line_id_y = 0;
    int line_id_uv = 1;
    int copy_size = w;
    if (top_h + 1 < h) {
        copy_size *= 2;
        line_id_y = 1;
    }
    int offset = 0;
    if (align) {
        global2share_copy_align(in_Y + top_h * w , y, copy_size);
    } else {
        offset = global2share_copy(in_Y + top_h * w , y, copy_size);
        y += offset;
    }
    if ((h_idx & 1) == 0) {
        copy_size = w;
        if ((top_h & 1) == 0 || top_h + 1 == h ) {
            line_id_uv = 0;
            copy_size = w >> 1;
        }
        if (align) {
            global2share_copy_align(in_U + (w >> 1) * (top_h >> 1), u, copy_size);
            global2share_copy_align(in_V + (w >> 1) * (top_h >> 1), v, copy_size);
        } else {
            offset = global2share_copy(in_U + (w >> 1) * (top_h >> 1), u, copy_size);
            u += offset;
            offset = global2share_copy(in_V + (w >> 1) * (top_h >> 1), v, copy_size);
            v += offset;
        }
    }
    __syncthreads();
    for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
        float src_w_idx = w_idx * t_w_trans;
        int left_w   = src_w_idx;
        int right_w  = MIN(w - 1, left_w + 1);
        int out_offset = h_idx * t_w + w_idx;
        half2 x1, x2;
        x1.x = y[left_w];
        x2.x = y[right_w];
        x1.y = y[left_w  + w * line_id_y];
        x2.y = y[right_w + w * line_id_y];

        half2 w_bi, alpha_h;
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left_w);
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out[out_offset] = __half2int_rn(
                (w_bi.y - w_bi.x) * __float2half_rn(src_h_idx - top_h) + w_bi.x);

        if((w_idx & 1) == 0 && (h_idx & 1) == 0) {
            int uv_offset = (h_idx >> 1) * (t_w >> 1) + (w_idx>>1);
            x1.x = u[left_w>>1];
            x2.x = u[right_w>>1];
            x1.y = u[line_id_uv * (w>>1) + (left_w>>1)];
            x2.y = u[line_id_uv * (w>>1) + (right_w>>1)];
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_U[uv_offset] = __half2int_rn(
                (w_bi.y - w_bi.x) * __float2half_rn(src_h_idx - top_h) + w_bi.x);
            x1.x = v[left_w>>1];
            x2.x = v[right_w>>1];
            x1.y = v[line_id_uv * (w>>1) + (left_w>>1)];
            x2.y = v[line_id_uv * (w>>1) + (right_w>>1)];
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_V[uv_offset] = __half2int_rn(
                (w_bi.y - w_bi.x) * __float2half_rn(src_h_idx - top_h) + w_bi.x);
        }
    }//end of w
}


#endif

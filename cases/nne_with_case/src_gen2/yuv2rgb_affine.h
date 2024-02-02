#ifndef YUV2RGB_AFFINE_H_
#define YUV2RGB_AFFINE_H_

__device__ __forceinline__ float clip(float value) {
  value += 0.5f;
  value = min(max(0.f,value),255.f);
  return value;
}

__device__ __forceinline__ float3 convert2rgb_TV_range(float y, float u, float v) {
    float3 tmp;
    y -= 16.f;
    y = max(0.f, y);
    y = 1.164f * y;
    u -= 128.f;
    v -= 128.f;
    tmp.x = clip(y + 1.596f * v);
    tmp.y = clip(y - 0.813f * v - 0.391f * u);
    tmp.z = clip(y + 2.018f * u);
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

template <bool full_range = false>
__global__ void yuv2rgb_affine_kernel(uint8_t *__restrict__ in_Y,
                                      uint8_t *__restrict__ in_U,
                                      uint8_t *__restrict__ in_V,
                                      uint8_t *__restrict__ output,
                                      int in_w, int in_h, int out_w, int out_h,
                                      float m0, float m1, float m2, float m3, float m4, float m5,
                                      int stride)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < out_w * out_h) {
        int hid = (i / stride * stride + i / out_w) % out_h;
        int wid = i % out_w;
        float fy = m3 * wid + m4 * hid + m5;
        int top = fy;
        if(top < 0 || top >= in_h) return;
        float fx = m0 * wid + m1 * hid + m2;
        int left = fx;
        if (left < 0 || left >= in_w) return;
        int src1_offset = top * in_w + left;
        int src2_offset = (top < in_h - 1) ? (src1_offset + in_w) : src1_offset;
        int edge_w = (left < in_w - 1) ? 1 : 0;

        float4 y, u, v;
        int idx = src1_offset;
        y.x = in_Y[idx];
        y.y = in_Y[idx + edge_w];
        idx = src2_offset;
        y.z = in_Y[idx];
        y.w = in_Y[idx + edge_w];

        int t0 = left>>1; 
        int t1 = (left+1)>>1; 
        src1_offset = (top>>1) * (in_w>>1);
        src2_offset = ((top + 1) >>1) * (in_w>>1);

        u.x = in_U[src1_offset + t0];
        u.y = in_U[src1_offset + t1];
        u.z = in_U[src2_offset + t0];
        u.w = in_U[src2_offset + t1];

        v.x = in_V[src1_offset + t0];
        v.y = in_V[src1_offset + t1];
        v.z = in_V[src2_offset + t0];
        v.w = in_V[src2_offset + t1];
        float3 a, b, c, d;
        if (full_range) {
            a = convert2rgb_full_range(y.x, u.x, v.x);
            b = convert2rgb_full_range(y.y, u.y, v.y);
            c = convert2rgb_full_range(y.z, u.z, v.z);
            d = convert2rgb_full_range(y.w, u.w, v.w);
        } else {
            a = convert2rgb_TV_range(y.x, u.x, v.x);
            b = convert2rgb_TV_range(y.y, u.y, v.y);
            c = convert2rgb_TV_range(y.z, u.z, v.z);
            d = convert2rgb_TV_range(y.w, u.w, v.w);
        }
        float x_diff = fx - left;
        float y_diff = fy - top;
        float scale1 = (1.f - x_diff) * (1.f - y_diff);
        float scale2 = x_diff * (1.f - y_diff);
        float scale3 = (1.f - x_diff) * y_diff;
        float scale4 = x_diff * y_diff;
        uchar3 rgb;
        rgb.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
        rgb.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
        rgb.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
        ((uchar3*)output)[hid * out_w + wid] = rgb;
    }
}

template <bool full_range = false>
__global__ void roi_nv122rgb_affine_norm_kernel(
    uint8_t *__restrict__ in_Y, uint8_t *__restrict__ in_UV, float *__restrict__ output,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
    float m0, float m1, float m2, float m3, float m4, float m5,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float pad1, float pad2, float pad3)
{
    int hid = blockIdx.x;
    for(int wid = threadIdx.x; wid < out_w; wid += blockDim.x) {
        float3 norm = {pad1, pad2, pad3};
        float fy = m3 * wid + m4 * hid + m5;
        int top = fy;
        float fx = m0 * wid + m1 * hid + m2;
        int left = fx;
        if(top >= 0 && top < roi_h && left >= 0 && left < roi_w) {
            float x_diff = fx - left;
            float y_diff = fy - top;
            top += roi_h_start;
            left += roi_w_start;
            int src1_offset = top * in_w + left;
            int src2_offset = (top < roi_h + roi_h_start - 1) ? (src1_offset + in_w) : src1_offset;
            int edge_w = (left < roi_w + roi_w_start - 1) ? 1 : 0;

            uchar4 y, u, v;
            y.x = in_Y[src1_offset];
            y.y = in_Y[src1_offset + edge_w];
            y.z = in_Y[src2_offset];
            y.w = in_Y[src2_offset + edge_w];

            int t0 = left & 0xfffffffe; 
            int t1 = left + 1 & 0xfffffffe; 
            src1_offset = (top>>1) * in_w;
            src2_offset = ((top + 1) >>1) * in_w;

            u.x = in_UV[src1_offset + t0];
            u.y = in_UV[src1_offset + t1];
            u.z = in_UV[src2_offset + t0];
            u.w = in_UV[src2_offset + t1];
            v.x = in_UV[src1_offset + t0 + 1];
            v.y = in_UV[src1_offset + t1 + 1];
            v.z = in_UV[src2_offset + t0 + 1];
            v.w = in_UV[src2_offset + t1 + 1];
            float3 a, b, c, d;
            if (full_range) {
                a = convert2rgb_full_range(y.x, u.x, v.x);
                b = convert2rgb_full_range(y.y, u.y, v.y);
                c = convert2rgb_full_range(y.z, u.z, v.z);
                d = convert2rgb_full_range(y.w, u.w, v.w);
            } else {
                a = convert2rgb_TV_range(y.x, u.x, v.x);
                b = convert2rgb_TV_range(y.y, u.y, v.y);
                c = convert2rgb_TV_range(y.z, u.z, v.z);
                d = convert2rgb_TV_range(y.w, u.w, v.w);
            }
            
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            uchar3 rgb;
            rgb.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            rgb.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            rgb.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            norm.x = (rgb.x * scale - mean1) * std1;
            norm.y = (rgb.y * scale - mean2) * std2;
            norm.z = (rgb.z * scale - mean3) * std3;
        }
        int out_idx = hid * out_w + wid;
        output[out_idx] = norm.x;
        output[out_idx + out_w * out_h] = norm.y;
        output[out_idx + (out_w << 1) * out_h] = norm.z;
    }
}

template <bool full_range = false, bool isRGB = true>
__global__ void roi_yu122rgb_affine_norm_kernel(
    uint8_t *__restrict__ in_Y,
    uint8_t *__restrict__ in_U,
    uint8_t *__restrict__ in_V, float *__restrict__ output,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
    float m0, float m1, float m2, float m3, float m4, float m5,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float pad1, float pad2, float pad3)
{
    int hid = blockIdx.x;
    for (int wid = threadIdx.x; wid < out_w; wid += blockDim.x)
    {
        float3 norm = {pad1, pad2, pad3};
        float fy = m3 * wid + m4 * hid + m5;
        int top = fy;
        float fx = m0 * wid + m1 * hid + m2;
        int left = fx;
        if(top >= 0 && top < roi_h && left >= 0 && left < roi_w) {
            float x_diff = fx - left;
            float y_diff = fy - top;
            top += roi_h_start;
            left += roi_w_start;
            int src1_offset = top * in_w + left;
            int src2_offset = (top < roi_h + roi_h_start - 1) ? (src1_offset + in_w) : src1_offset;
            int edge_w = (left < roi_w + roi_w_start - 1) ? 1 : 0;

            uchar4 y, u, v;
            y.x = in_Y[src1_offset];
            y.y = in_Y[src1_offset + edge_w];
            y.z = in_Y[src2_offset];
            y.w = in_Y[src2_offset + edge_w];

            int t0 = left>>1; 
            int t1 = (left+1)>>1; 
            src1_offset = (top>>1) * (in_w>>1);
            src2_offset = ((top + 1) >>1) * (in_w>>1);
            u.x = in_U[src1_offset + t0];
            u.y = in_U[src1_offset + t1];
            u.z = in_U[src2_offset + t0];
            u.w = in_U[src2_offset + t1];

            v.x = in_V[src1_offset + t0];
            v.y = in_V[src1_offset + t1];
            v.z = in_V[src2_offset + t0];
            v.w = in_V[src2_offset + t1];
            float3 a, b, c, d;
            if (full_range) {
                a = convert2rgb_full_range(y.x, u.x, v.x);
                b = convert2rgb_full_range(y.y, u.y, v.y);
                c = convert2rgb_full_range(y.z, u.z, v.z);
                d = convert2rgb_full_range(y.w, u.w, v.w);
            } else {
                a = convert2rgb_TV_range(y.x, u.x, v.x);
                b = convert2rgb_TV_range(y.y, u.y, v.y);
                c = convert2rgb_TV_range(y.z, u.z, v.z);
                d = convert2rgb_TV_range(y.w, u.w, v.w);
            }
            
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            uchar3 rgb;
            rgb.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            rgb.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            rgb.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (isRGB) {
                norm.x = (rgb.x * scale - mean1) * std1;
                norm.y = (rgb.y * scale - mean2) * std2;
                norm.z = (rgb.z * scale - mean3) * std3;
            } else {
                norm.x = (rgb.x * scale - mean3) * std3;
                norm.y = (rgb.y * scale - mean2) * std2;
                norm.z = (rgb.z * scale - mean1) * std1;
            }
        }
        int out_idx = hid * out_w + wid;
        if (isRGB) {
            output[out_idx] = norm.x;
            output[out_idx + out_w * out_h] = norm.y;
            output[out_idx + (out_w << 1) * out_h] = norm.z;
        } else {
            output[out_idx] = norm.z;
            output[out_idx + out_w * out_h] = norm.y;
            output[out_idx + (out_w << 1) * out_h] = norm.x;
        }
    }
}
#endif

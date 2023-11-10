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

template<bool full_range = false>
__global__ void yuv2rgb_affine_kernel(uint8_t* __restrict__ input, uint8_t* __restrict__ output,
        int in_w, int in_h, int out_w, int out_h,
        float m0, float m1, float m2, float m3, float m4, float m5) {
    extern __shared__ int sm[];
    int*  sm_base = sm + out_w * 12;
    int*  sm_flg  = sm_base + out_w;
    uint8_t* sm_data = (uint8_t*)(sm + out_w * 13 + 1);
    if (threadIdx.x == 0) {
        *sm_flg = 0;
    }
    __syncthreads();
    int y = blockIdx.x;
    for(int x = threadIdx.x; x < out_w; x += blockDim.x) {
        int top_h = m3 * x + m4 * y + m5;
        if(top_h < 0 || top_h >= in_h)  continue;
        int left_w = m0 * x + m1 * y + m2;
        if (left_w < 0 || left_w >= in_w) continue;
        int src1_offset = top_h * in_w + left_w;
        int src2_offset = (top_h < in_h - 1) ? (src1_offset + in_w) : src1_offset;
        int edge_w = (left_w < in_w - 1) ? 1 : 0;
        int st_base = atomicAdd(sm_flg, 12);   
        sm_base[x] = st_base;
        int2 addr;
        int2 *tmp_ptr = (int2*)(sm + st_base);
        addr.x = src1_offset;   
        addr.y = src1_offset + edge_w;
        tmp_ptr[0] = addr;
        addr.x = src2_offset;
        addr.y = src2_offset + edge_w;
        tmp_ptr[1] = addr;

        int t0 = left_w>>1; 
        int t1 = (left_w+1)>>1; 
        src1_offset = (top_h>>1) * (in_w>>1) + in_w * in_h;
        src2_offset = ((top_h + 1) >>1) * (in_w>>1) + in_w * in_h;
        addr.x = src1_offset + t0;
        addr.y = src1_offset + t1;
        tmp_ptr[2] = addr;
        addr.x = src2_offset + t0;   
        addr.y = src2_offset + t1;
        tmp_ptr[3] = addr;

        src1_offset += in_w * in_h >> 2;
        src2_offset += in_w * in_h >> 2;
        addr.x = src1_offset + t0;
        addr.y = src1_offset + t1;
        tmp_ptr[4] = addr;
        addr.x = src2_offset + t0;   
        addr.y = src2_offset + t1;
        tmp_ptr[5] = addr;
    }
    __syncthreads();

    for(int x = threadIdx.x; x < *sm_flg; x+= blockDim.x) {
        sm_data[x] = input[sm[x]];
    }
    __syncthreads();

    for (int x = threadIdx.x; x < out_w; x += blockDim.x) {
        float fy = m3 * x + m4 * y + m5;
        int top_h = fy;
        if(top_h < 0 || top_h >= in_h) continue;
        float fx = m0 * x + m1 * y + m2;
        int left_w = fx;
        if (left_w < 0 || left_w >= in_w) continue;

        int base = sm_base[x]; 
        uint8_t *tmp_ptr = (uint8_t*)(sm_data + base);
        uint8_t y1 = tmp_ptr[0];
        uint8_t y2 = tmp_ptr[1];
        uint8_t y3 = tmp_ptr[2];
        uint8_t y4 = tmp_ptr[3];
        uint8_t u1 = tmp_ptr[4];
        uint8_t u2 = tmp_ptr[5];
        uint8_t u3 = tmp_ptr[6];
        uint8_t u4 = tmp_ptr[7];
        uint8_t v1 = tmp_ptr[8];
        uint8_t v2 = tmp_ptr[9];
        uint8_t v3 = tmp_ptr[10];
        uint8_t v4 = tmp_ptr[11];
        float3 a, b, c, d;
        if (full_range) {
            a = convert2rgb_full_range(y1, u1, v1);
            b = convert2rgb_full_range(y2, u2, v2);
            c = convert2rgb_full_range(y3, u3, v3);
            d = convert2rgb_full_range(y4, u4, v4);
        } else {
            a = convert2rgb_TV_range(y1, u1, v1);
            b = convert2rgb_TV_range(y2, u2, v2);
            c = convert2rgb_TV_range(y3, u3, v3);
            d = convert2rgb_TV_range(y4, u4, v4);
        }

        float x_diff = fx - left_w;
        float y_diff = fy - top_h;
        float scale1 = (1.f - x_diff) * (1.f - y_diff);
        float scale2 = x_diff * (1.f - y_diff);
        float scale3 = (1.f - x_diff) * y_diff;
        float scale4 = x_diff * y_diff;
        uchar3 rgb;
        rgb.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
        rgb.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
        rgb.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
        ((uchar3*)output)[y * out_w + x] = rgb;
    }
}

template<bool full_range = false>
__global__ void roi_nv122rgb_affine_norm_kernel(uint8_t* __restrict__ input, float* __restrict__ output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
        float m0, float m1, float m2, float m3, float m4, float m5,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3) {
    extern __shared__ int sm[];
    int rows = blockDim.y;
    int*  sm_base = sm + out_w * rows * 12;
    int*  sm_flg  = sm_base + out_w * rows;
    uint8_t* sm_data = (uint8_t*)(sm + out_w * rows * 13 + rows);
    if (threadIdx.x < rows) {
        sm_flg[threadIdx.x] = 0;
    }
    __syncthreads();
    int y = blockIdx.y * rows + threadIdx.y;
    for(int x = threadIdx.x; x < out_w; x += blockDim.x) {
        int top_h = m3 * x + m4 * y + m5;
        if(top_h < 0 || top_h >= roi_h)  continue;
        int left_w = m0 * x + m1 * y + m2;
        if (left_w < 0 || left_w >= roi_w) continue;
        top_h += roi_h_start;
        left_w += roi_w_start;
        int src1_offset = top_h * in_w + left_w;
        int src2_offset = (top_h < roi_h + roi_h_start - 1) ? (src1_offset + in_w) : src1_offset;
        int edge_w = (left_w < roi_w + roi_w_start - 1) ? 1 : 0;
        int st_base = atomicAdd(sm_flg + threadIdx.y, 12);   
        sm_base[x + threadIdx.y * out_w] = st_base;
        int2 addr;
        int2 *tmp_ptr = (int2*)(sm + st_base + threadIdx.y * out_w * 12);
        addr.x = src1_offset;   
        addr.y = src1_offset + edge_w;
        tmp_ptr[0] = addr;
        addr.x = src2_offset;
        addr.y = src2_offset + edge_w;
        tmp_ptr[1] = addr;

        int t0 = left_w & 0xfffffffe; 
        int t1 = left_w+1 & 0xfffffffe; 
        src1_offset = (top_h>>1) * in_w + in_w * in_h;
        src2_offset = ((top_h + 1) >>1) * in_w + in_w * in_h;
        addr.x = src1_offset + t0;
        addr.y = src1_offset + t1;
        tmp_ptr[2] = addr;
        tmp_ptr[4] = {addr.x + 1, addr.y + 1};
        addr.x = src2_offset + t0;   
        addr.y = src2_offset + t1;
        tmp_ptr[3] = addr;
        tmp_ptr[5] = {addr.x + 1, addr.y + 1};
    }
    __syncthreads();

    for(int x = threadIdx.x; x < sm_flg[threadIdx.y]; x+= blockDim.x) {
        sm_data[x + threadIdx.y * out_w * 12] = input[sm[x + threadIdx.y * out_w * 12]];
    }
    __syncthreads();

    for (int x = threadIdx.x; x < out_w; x += blockDim.x) {
        float3 norm = {pad1, pad2, pad3};
        float fy = m3 * x + m4 * y + m5;
        int top_h = fy;
        float fx = m0 * x + m1 * y + m2;
        int left_w = fx;
        if(top_h >= 0 && top_h < roi_h && left_w >= 0 && left_w < roi_w) {
            int base = sm_base[x + threadIdx.y * out_w]; 
            uint8_t *tmp_ptr = (uint8_t*)(sm_data + base + threadIdx.y * out_w * 12);
            uint8_t y1 = tmp_ptr[0];
            uint8_t y2 = tmp_ptr[1];
            uint8_t y3 = tmp_ptr[2];
            uint8_t y4 = tmp_ptr[3];
            uint8_t u1 = tmp_ptr[4];
            uint8_t u2 = tmp_ptr[5];
            uint8_t u3 = tmp_ptr[6];
            uint8_t u4 = tmp_ptr[7];
            uint8_t v1 = tmp_ptr[8];
            uint8_t v2 = tmp_ptr[9];
            uint8_t v3 = tmp_ptr[10];
            uint8_t v4 = tmp_ptr[11];
            float3 a, b, c, d;
            if (full_range) {
                a = convert2rgb_full_range(y1, u1, v1);
                b = convert2rgb_full_range(y2, u2, v2);
                c = convert2rgb_full_range(y3, u3, v3);
                d = convert2rgb_full_range(y4, u4, v4);
            } else {
                a = convert2rgb_TV_range(y1, u1, v1);
                b = convert2rgb_TV_range(y2, u2, v2);
                c = convert2rgb_TV_range(y3, u3, v3);
                d = convert2rgb_TV_range(y4, u4, v4);
            }

            float x_diff = fx - left_w;
            float y_diff = fy - top_h;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 rgb;
            rgb.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            rgb.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            rgb.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            norm.x = (rgb.x * scale - mean1) * std1;
            norm.y = (rgb.y * scale - mean2) * std2;
            norm.z = (rgb.z * scale - mean3) * std3;
        }
        int out_idx = y * out_w + x;
        output[out_idx] = norm.x;
        output[out_idx + out_w * out_h] = norm.y;
        output[out_idx + (out_w << 1) * out_h] = norm.z;
    }
}

template<bool full_range = false, bool isRGB = true>
__global__ void roi_yu122rgb_affine_norm_kernel(uint8_t* __restrict__ input, float* __restrict__ output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
        float m0, float m1, float m2, float m3, float m4, float m5,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3) {
    extern __shared__ int sm[];
    const int rows = blockDim.y;
    int*  sm_base = sm + out_w * rows * 12;
    int*  sm_flg  = sm_base + out_w * rows;
    uint8_t* sm_data = (uint8_t*)(sm + out_w * rows * 13 + rows);
    if (threadIdx.x < rows) {
        sm_flg[threadIdx.x] = 0;
    }
    __syncthreads();
    int y = blockIdx.y * rows + threadIdx.y;
    for(int x = threadIdx.x; x < out_w; x += blockDim.x) {
        int top_h = m3 * x + m4 * y + m5;
        if(top_h < 0 || top_h >= roi_h)  continue;
        int left_w = m0 * x + m1 * y + m2;
        if (left_w < 0 || left_w >= roi_w) continue;
        top_h += roi_h_start;
        left_w += roi_w_start;
        int src1_offset = top_h * in_w + left_w;
        int src2_offset = (top_h < roi_h + roi_h_start - 1) ? (src1_offset + in_w) : src1_offset;
        int edge_w = (left_w < roi_w + roi_w_start - 1) ? 1 : 0;
        int st_base = atomicAdd(sm_flg + threadIdx.y, 12);   
        sm_base[x + threadIdx.y * out_w] = st_base;
        int2 addr;
        int2 *tmp_ptr = (int2*)(sm + st_base + threadIdx.y * out_w * 12);
        addr.x = src1_offset;   
        addr.y = src1_offset + edge_w;
        tmp_ptr[0] = addr;
        addr.x = src2_offset;
        addr.y = src2_offset + edge_w;
        tmp_ptr[1] = addr;

        int t0 = left_w>>1; 
        int t1 = (left_w+1)>>1; 
        src1_offset = (top_h>>1) * (in_w>>1) + in_w * in_h;
        src2_offset = ((top_h + 1) >>1) * (in_w>>1) + in_w * in_h;
        addr.x = src1_offset + t0;
        addr.y = src1_offset + t1;
        tmp_ptr[2] = addr;
        addr.x = src2_offset + t0;   
        addr.y = src2_offset + t1;
        tmp_ptr[3] = addr;

        src1_offset += in_w * in_h >> 2;
        src2_offset += in_w * in_h >> 2;
        addr.x = src1_offset + t0;
        addr.y = src1_offset + t1;
        tmp_ptr[4] = addr;
        addr.x = src2_offset + t0;   
        addr.y = src2_offset + t1;
        tmp_ptr[5] = addr;
    }
    __syncthreads();

    for(int x = threadIdx.x; x < sm_flg[threadIdx.y]; x+= blockDim.x) {
        sm_data[x + threadIdx.y * out_w * 12] = input[sm[x + threadIdx.y * out_w * 12]];
    }
    __syncthreads();

    for (int x = threadIdx.x; x < out_w; x += blockDim.x) {
        float3 norm = {pad1, pad2, pad3};
        float fy = m3 * x + m4 * y + m5;
        int top_h = fy;
        float fx = m0 * x + m1 * y + m2;
        int left_w = fx;
        if(top_h >= 0 && top_h < roi_h && left_w >= 0 && left_w < roi_w) {
            int base = sm_base[x + threadIdx.y * out_w]; 
            uint8_t *tmp_ptr = (uint8_t*)(sm_data + base + threadIdx.y * out_w * 12);
            uint8_t y1 = tmp_ptr[0];
            uint8_t y2 = tmp_ptr[1];
            uint8_t y3 = tmp_ptr[2];
            uint8_t y4 = tmp_ptr[3];
            uint8_t u1 = tmp_ptr[4];
            uint8_t u2 = tmp_ptr[5];
            uint8_t u3 = tmp_ptr[6];
            uint8_t u4 = tmp_ptr[7];
            uint8_t v1 = tmp_ptr[8];
            uint8_t v2 = tmp_ptr[9];
            uint8_t v3 = tmp_ptr[10];
            uint8_t v4 = tmp_ptr[11];
            float3 a, b, c, d;
            if (full_range) {
                a = convert2rgb_full_range(y1, u1, v1);
                b = convert2rgb_full_range(y2, u2, v2);
                c = convert2rgb_full_range(y3, u3, v3);
                d = convert2rgb_full_range(y4, u4, v4);
            } else {
                a = convert2rgb_TV_range(y1, u1, v1);
                b = convert2rgb_TV_range(y2, u2, v2);
                c = convert2rgb_TV_range(y3, u3, v3);
                d = convert2rgb_TV_range(y4, u4, v4);
            }

            float x_diff = fx - left_w;
            float y_diff = fy - top_h;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 rgb;
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
        int out_idx = y * out_w + x;
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

#ifndef RGB_AFFINE_H_
#define RGB_AFFINE_H_

template<bool IsRGB>
__global__ void rgb24_affine_kernel(uint8_t* __restrict__ input, uint8_t* __restrict__ output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
        float m0, float m1, float m2, float m3, float m4, float m5) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = blockIdx.y;
    float fy = m3 * x + m4 * y + m5;
    float fx = m0 * x + m1 * y + m2;
    int top = fy;
    int left = fx;
    bool w_run = top >= 0 && top < roi_h && left >= 0 && left < roi_w;
    if (w_run) {
        uchar3 rgb;
        uchar4 r, g, b;
        int src1_offset = (top + roi_h_start) * in_w + left + roi_w_start;
        int src2_offset = (top < roi_h - 1) ? (src1_offset + in_w) : src1_offset;
        int edge_w = (left < roi_w - 1) ? 1 : 0;
        if (IsRGB) {
            src1_offset *= 3;
            src2_offset *= 3;
            edge_w *= 3;
        }
        int idx = src1_offset + 0;
        r.x = input[idx];
        r.y = input[idx + edge_w];
        idx = src2_offset + 0;
        r.z = input[idx];
        r.w = input[idx + edge_w];
        if (IsRGB) {
            idx = src1_offset + 1;
            g.x = input[idx];
            g.y = input[idx + edge_w];
            idx = src2_offset + 1;
            g.z = input[idx];
            g.w = input[idx + edge_w];
            idx = src1_offset + 2;
            b.x = input[idx];
            b.y = input[idx + edge_w];
            idx = src2_offset + 2;
            b.z = input[idx];
            b.w = input[idx + edge_w];
        }
        float x_diff = fx - left;
        float y_diff = fy - top;
        float scale1 = (1.f - x_diff) * (1.f - y_diff);
        float scale2 = x_diff * (1.f - y_diff);
        float scale3 = (1.f - x_diff) * y_diff;
        float scale4 = x_diff * y_diff;

        rgb.x = r.x * scale1 + r.y * scale2 + r.z * scale3 + r.w * scale4;
        if (IsRGB) {
            rgb.y = g.x * scale1 + g.y * scale2 + g.z * scale3 + g.w * scale4;
            rgb.z = b.x * scale1 + b.y * scale2 + b.z * scale3 + b.w * scale4;
        }
        int out_idx = y * out_w + x;
        if (IsRGB) {
            ((uchar3*)output)[out_idx] = rgb;
        } else {
            output[out_idx] = rgb.x;
        }
    }
}

__global__ void batch_rgb24_affine_kernel(uint8_t* __restrict__ input, uint8_t* __restrict__ output, int out_batch_offset,
        int in_w, int in_h, RGBAffineInfo_t *param) {
    __shared__ RGBAffineInfo_t sm;
    if (threadIdx.x == 0) {
        sm = param[blockIdx.x];
    }
    __syncthreads();
    output += out_batch_offset * blockIdx.x;
    int roi_w_start = sm.roi_w_start;
    int roi_h_start = sm.roi_h_start;
    int roi_w = sm.roi_w;
    int roi_h = sm.roi_h;
    int out_w = sm.out_w;
    int out_h = sm.out_h;
    float m0 = sm.m[0];
    float m1 = sm.m[1];
    float m2 = sm.m[2];
    float m3 = sm.m[3];
    float m4 = sm.m[4];
    float m5 = sm.m[5];
    for(int y = blockIdx.y; y < out_h; y += gridDim.y) {
        for(int x = threadIdx.x; x < out_w; x += blockDim.x) {
            float fy = m3 * x + m4 * y + m5;
            int top = fy;
            if(top >= 0 && top < roi_h) {
                float fx = m0 * x + m1 * y + m2;
                int left = fx;
                if (left < 0 || left >= roi_w) continue;
                int src1_offset = (top + roi_h_start) * in_w + (left + roi_w_start);
                int src2_offset = (top < roi_h - 1) ? (src1_offset + in_w) : src1_offset;
                int edge_w = (left < roi_w - 1) ? 1 : 0;
                uchar3 rgb;
                uchar4 r, g, b;
                src1_offset *= 3;
                src2_offset *= 3;
                edge_w *= 3;
                int idx = src1_offset + 0;
                r.x = input[idx];
                r.y = input[idx + edge_w];
                idx = src2_offset + 0;
                r.z = input[idx];
                r.w = input[idx + edge_w];
                idx = src1_offset + 1;
                g.x = input[idx];
                g.y = input[idx + edge_w];
                idx = src2_offset + 1;
                g.z = input[idx];
                g.w = input[idx + edge_w];
                idx = src1_offset + 2;
                b.x = input[idx];
                b.y = input[idx + edge_w];
                idx = src2_offset + 2;
                b.z = input[idx];
                b.w = input[idx + edge_w];
                float x_diff = fx - left;
                float y_diff = fy - top;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;

                rgb.x = r.x * scale1 + r.y * scale2 + r.z * scale3 + r.w * scale4;
                rgb.y = g.x * scale1 + g.y * scale2 + g.z * scale3 + g.w * scale4;
                rgb.z = b.x * scale1 + b.y * scale2 + b.z * scale3 + b.w * scale4;
                int out_idx = y * out_w + x;
                ((uchar3*)output)[out_idx] = rgb;
            }
        }
    }
}
#endif

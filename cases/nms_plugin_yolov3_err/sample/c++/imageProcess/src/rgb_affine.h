#ifndef RGB_AFFINE_H_
#define RGB_AFFINE_H_


__global__ void rgb24_affine_kernel(uint8_t* __restrict__ input, uint8_t* __restrict__ output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
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
        if(top_h >= 0 && top_h < roi_h) {
          int left_w = m0 * x + m1 * y + m2;
          if (left_w < 0 || left_w >= roi_w) continue;
          int src1_offset = (top_h + roi_h_start) * in_w * 3 + 3 * (left_w + roi_w_start);
          int src2_offset = (top_h < roi_h - 1) ? (src1_offset + in_w * 3) : src1_offset;
          int edge_w = (left_w < roi_w - 1) ? 3 : 0;
          int st_base = atomicAdd(sm_flg, 12);   
          sm_base[x] = st_base;
          int2 addr;
          int2 *tmp_ptr = (int2*)(sm + st_base);
          addr.x = src1_offset;   
          addr.y = src1_offset + 1;
          tmp_ptr[0] = addr;
          addr.x = src1_offset + 2;
          addr.y = src1_offset + 3;
          tmp_ptr[1] = addr;
          addr.x = src1_offset + 4;
          addr.y = src1_offset + 5;
          tmp_ptr[2] = addr;
          addr.x = src2_offset;   
          addr.y = src2_offset + 1;
          tmp_ptr[3] = addr;
          addr.x = src2_offset + 2;
          addr.y = src2_offset + 3;
          tmp_ptr[4] = addr;
          addr.x = src2_offset + 4;
          addr.y = src2_offset + 5;
          tmp_ptr[5] = addr;
        }
    }
   __syncthreads();

   for(int x = threadIdx.x; x < *sm_flg; x+= blockDim.x) {
      sm_data[x] = input[sm[x]];
   }
   __syncthreads();

   for (int x = threadIdx.x; x < out_w; x += blockDim.x) {
        float fy = m3 * x + m4 * y + m5;
        int top_h = fy;
        if(top_h >= 0 && top_h < roi_h) {
            float fx = m0 * x + m1 * y + m2;
            int left_w = fx;
          if (left_w < 0 || left_w >= roi_w) continue;

            int base = sm_base[x]; 
            uint8_t *tmp_ptr = (uint8_t*)(sm_data + base);
            uint8_t a0 = tmp_ptr[0];
            uint8_t a1 = tmp_ptr[1];
            uint8_t a2 = tmp_ptr[2];
            uint8_t a3 = tmp_ptr[3];
            uint8_t a4 = tmp_ptr[4];
            uint8_t a5 = tmp_ptr[5];
            uint8_t b0 = tmp_ptr[6];
            uint8_t b1 = tmp_ptr[7];
            uint8_t b2 = tmp_ptr[8];
            uint8_t b3 = tmp_ptr[9];
            uint8_t b4 = tmp_ptr[10];
            uint8_t b5 = tmp_ptr[11];
            float x_diff = fx - left_w;
            float y_diff = fy - top_h;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            uchar3 rgb;
            rgb.x = a0 * scale1 + a3 * scale2 + b0 * scale3 + b3 * scale4;
            rgb.y = a1 * scale1 + a4 * scale2 + b1 * scale3 + b4 * scale4;
            rgb.z = a2 * scale1 + a5 * scale2 + b2 * scale3 + b5 * scale4;
            ((uchar3*)output)[y * out_w + x] = rgb;
        }
    }
}

__global__ void gray_affine_kernel(uint8_t* __restrict__ input, uint8_t* __restrict__ output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
        float m0, float m1, float m2, float m3, float m4, float m5) {
    int y = blockIdx.x;
    for (int x = threadIdx.x; x < out_w; x += blockDim.x) {
        float fx = m0 * x + m1 * y + m2;
        float fy = m3 * x + m4 * y + m5;
        int in_x_idx = fx;
        int in_y_idx = fy;
        uint8_t out_tmp = 0;
        if (in_y_idx >= 0 && in_y_idx < roi_h) {
            if (in_x_idx >= 0 && in_x_idx < roi_w) {
                int src1_offset = (in_y_idx + roi_h_start) * in_w + (in_x_idx + roi_w_start);
                int src2_offset = (in_y_idx < roi_h - 1) ? (src1_offset + in_w) : src1_offset;
                int edge_w = (in_x_idx < roi_w - 1) ? 1 : 0;
                uchar a = input[src1_offset];
                uchar b = input[src1_offset + edge_w];
                uchar c = input[src2_offset];
                uchar d = input[src2_offset + edge_w];

                float x_diff = fx - in_x_idx;
                float y_diff = fy - in_y_idx;
                float scale1 = (1.f - x_diff) * (1.f - y_diff);
                float scale2 = x_diff * (1.f - y_diff);
                float scale3 = (1.f - x_diff) * y_diff;
                float scale4 = x_diff * y_diff;
                out_tmp = a * scale1 + b * scale2 + c * scale3 + d * scale4;
            }
        }
        output[y * out_w + x] = out_tmp;
     }
 }


#endif

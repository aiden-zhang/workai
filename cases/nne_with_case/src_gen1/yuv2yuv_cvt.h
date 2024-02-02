#ifndef YUV2YUV_CVT_H_
#define YUV2YUV_CVT_H_


template <typename IN, typename OUT, int size>
__global__ void __launch_bounds__(512) yuv444p_to_yuv420p(IN* __restrict__ sy, IN* __restrict__ su, IN* __restrict__ sv, 
    IN* __restrict__ dy, OUT* __restrict__ du, OUT* __restrict__ dv, int w, int h, int stride) {
    int w_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int h_idx = threadIdx.y + blockDim.y * blockIdx.y;
    if (w_idx < w && h_idx < h) {
        IN su1 = su[(h_idx * 2) * stride + w_idx];
        IN su2 = su[(h_idx * 2 + 1) * stride + w_idx];
        IN sv1 = sv[(h_idx * 2) * stride + w_idx];
        IN sv2 = sv[(h_idx * 2 + 1) * stride + w_idx];
        uint8_t out_u[size];
        uint8_t out_v[size];
        for (int i = 0; i < size; ++i) {
            short sum_u = 0;
            short sum_v = 0;
            uchar2 t = ((uchar2*)&su1)[i];
            sum_u = sum_u + t.x + t.y;
            t = ((uchar2*)&su2)[i];
            sum_u = sum_u + t.x + t.y;
            t = ((uchar2*)&sv1)[i];
            sum_v = sum_v + t.x + t.y;
            t = ((uchar2*)&sv2)[i];
            sum_v = sum_v + t.x + t.y;
            out_u[i] = sum_u >> 2;
            out_v[i] = sum_v >> 2;
        }
        dy[(h_idx * 2) * w + w_idx] = sy[(h_idx * 2) * stride + w_idx];
        dy[(h_idx * 2 + 1) * w + w_idx] = sy[(h_idx * 2 + 1) * stride + w_idx];
        du[h_idx * w + w_idx] = *(OUT*)out_u;
        dv[h_idx * w + w_idx] = *(OUT*)out_v;
    }
}

template <typename IN, typename OUT, int size>
__global__ void yuv440p_to_yuv420p(IN* __restrict__ sy, IN* __restrict__ su, IN* __restrict__ sv, 
    IN* __restrict__ dy, OUT* __restrict__ du, OUT* __restrict__ dv, int w, int h, int stride) {
    int w_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int h_idx = threadIdx.y + blockDim.y * blockIdx.y;
    if (w_idx < w && h_idx < h) {
        IN su1 = su[h_idx * stride + w_idx];
        IN sv1 = sv[h_idx * stride + w_idx];
        uint8_t out_u[size];
        uint8_t out_v[size];
        for (int i = 0; i < size; ++i) {
            uchar2 t = ((uchar2*)&su1)[i];
            out_u[i] = ((short)t.x + t.y) >> 1;
            t = ((uchar2*)&sv1)[i];
            out_v[i] = ((short)t.x + t.y) >> 1;
        }
        dy[(h_idx * 2) * w + w_idx] = sy[(h_idx * 2) * stride + w_idx];
        dy[(h_idx * 2 + 1) * w + w_idx] = sy[(h_idx * 2 + 1) * stride + w_idx];
        du[h_idx * w + w_idx] = *(OUT*)out_u;
        dv[h_idx * w + w_idx] = *(OUT*)out_v;
    }
}


#endif

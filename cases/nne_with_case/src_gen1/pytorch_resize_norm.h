#ifndef PYTORCH_RESIZE_NORM_H_
#define PYTORCH_RESIZE_NORM_H_

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
            uchar4 t = data[xmin + STEP];
			ss->x += t.x * coeff;
			ss->y += t.y * coeff;
			ss->z += t.z * coeff;
		}

		horiz_mad_coeffs_step(ss, data, k, xmin, in_size, Int2Type<STEP + 1>());
	}
    template <int STEP>
    __device__ __forceinline__ void horiz_mad_coeffs_step(
            int3* ss, uchar3* data, float* k, int xmin, int in_size, Int2Type<STEP>)
    {
        if (xmin + STEP < in_size) {
            int coeff = (int)((k[STEP] < 0.f ? -0.5f : 0.5f)  + k[STEP] * (1 << PRECISION_BITS));
            uchar3 t = data[xmin + STEP];
            ss->x += t.x * coeff;
            ss->y += t.y * coeff;
            ss->z += t.z * coeff;
        }

        horiz_mad_coeffs_step(ss, data, k, xmin, in_size, Int2Type<STEP + 1>());
    }

	__device__ __forceinline__ void horiz_mad_coeffs_step(
			int3* ss, uchar4* data, float* k, int xmin, int in_size, Int2Type<STEPS>)
	{
	   return;
	}

    __device__ __forceinline__ void horiz_mad_coeffs_step(
            int3* ss, uchar3* data, float* k, int xmin, int in_size, Int2Type<STEPS>)
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
    uint8_t *u1 = y1 + (roi_w & 0xfffffff8) + 16;
    uint8_t *v1 = u1 + ((roi_w + 1) / 2 & 0xfffffff8) + 16;
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
        int out_idx = blockIdx.x * out_w + w_idx;
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

template <int KSIZE = 1, bool align = false>
__global__ void rgba_horizontal_resize_kernel(uint8_t* __restrict__ in, uchar4* __restrict__ out,
        int in_w, int in_h, int out_w, int out_h, float filter_scale) {
    typedef KernelCoeff<KSIZE> KernelCoeffT;
    KernelCoeffT kernel_coeff = KernelCoeffT();
    int horiz_bounds[1];
    float horiz_coeffs[KSIZE];
    extern __shared__ uint8_t sm[];
    uint8_t *rgb = sm;
    int h_idx = blockIdx.y;
    if (align) {
        global2share_copy_align(in + h_idx * in_w * 3, rgb, in_w * 3);
    } else {
        int sm_offset = global2share_copy(in + h_idx * in_w * 3, rgb, in_w * 3);
        rgb += sm_offset;
    }
    __syncthreads();

    for (int w_idx = threadIdx.x; w_idx < out_w; w_idx += blockDim.x) {
        kernel_coeff.compute_coeffs(w_idx, in_w, filter_scale, horiz_coeffs, horiz_bounds);
        if (w_idx < out_w) {
            int xmin = horiz_bounds[0];
            int3 ss = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1)};
            kernel_coeff.horiz_mad_coeffs_step(&ss, (uchar3*)rgb, horiz_coeffs, horiz_bounds[0], in_w, Int2Type<0>());

            int out_idx = h_idx * out_w + w_idx;
            out[out_idx] = {clip8(ss.x), clip8(ss.y), clip8(ss.z), 0};
        }
    }
}


template <bool bgr_format = false, bool norm = true, int KSIZE=1>
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

template <bool bgr_format = false, bool norm = true, int KSIZE=1>
__global__ void rgba_vertical_resize_crop_norm_kernel(uchar4* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int out_w, int out_h, int crop_start_w, int crop_start_h,
    float filter_scale, float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale) {
    typedef KernelCoeff<KSIZE> KernelCoeffT;
    KernelCoeffT kernel_coeff = KernelCoeffT();
    float coeffs[KSIZE];
    int bounds[1];

    int h_idx = blockIdx.y;
    for (int tw_idx = threadIdx.x; tw_idx < out_w; tw_idx += blockDim.x) {
        kernel_coeff.compute_coeffs(h_idx + crop_start_h, in_h, filter_scale, coeffs, bounds);
        int3 ss = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1)};
        kernel_coeff.vert_mad_coeffs_step(&ss, in, coeffs, tw_idx + crop_start_w, bounds[0], in_h, in_w, Int2Type<0>());

        float3 out_tmp;
        out_tmp.x = clip8(ss.x);
        out_tmp.y = clip8(ss.y);
        out_tmp.z = clip8(ss.z);

        float3 norm_tmp = {0, 0, 0};
        if (bgr_format) {
            if (norm) {
                norm_tmp.z = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.x = (out_tmp.z * scale - mean3) * std3;
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

#include "cuda_fp16.h"
#include "cuda_runtime.h"


__device__ __forceinline__ half2 short2half2(int16_t uv)
{
   half2 ret;
   //v1. ptr cast
   //v2. bit cast
   ret.x = ((int8_t*)&uv)[0];
   ret.y = ((int8_t*)&uv)[1];
   return ret;
}


template<bool swizzling>
__device__ __forceinline__ half2 cvt2half2_low(uint32_t y)
{
  half2 ret; 
  if(swizzling)
  {
    //1,3 
    ret.x = ((uint8_t*)&y)[0];
    ret.y = ((uint8_t*)&y)[2];
  }
  else
  {
    //1,2
    ret.x = ((uint8_t*)&y)[0];
    ret.y = ((uint8_t*)&y)[1];
  }
  ret = __hmul2(ret,__float2half2_rn(1.164f));
  return ret;
}

template<bool swizzling>
__device__ __forceinline__ half2 cvt2half2_high(uint32_t y)
{
  half2 ret;
  if(swizzling)
  {
    //2,4
    ret.x = ((uint8_t*)&y)[1];
    ret.y = ((uint8_t*)&y)[3];
  }
  else
  {
    //3,4
    ret.x = ((uint8_t*)&y)[2];
    ret.y = ((uint8_t*)&y)[3];
  }
  ret = __hmul2(ret,__float2half2_rn(1.164f));
  return ret;
}

__device__ __forceinline__ uint32_t cvt2r_4x(half2 y13,half2 y24,half2 v1v2)
{
   half2 r13 = __hfma2(__float2half2_rn(1.596f),v1v2,y13);
   half2 r24 = __hfma2(__float2half2_rn(1.596f),v1v2,y24);
   half2 u8_max = __float2half2_rn(255.f);
   half2 u8_min = __float2half2_rn(0.f);
   r13 = __hmax2(__hmin2(u8_max,r13),u8_min);
   r24 = __hmax2(__hmin2(u8_max,r24),u8_min);
   uint32_t ret;
   ((uint8_t*)&ret)[0] = ushort(r13.x);
   ((uint8_t*)&ret)[1] = ushort(r24.x);
   ((uint8_t*)&ret)[2] = ushort(r13.y);
   ((uint8_t*)&ret)[3] = ushort(r24.y);
   return ret;
}

__device__ __forceinline__ uint32_t cvt2b_4x(half2 y13,half2 y24,half2 u1u2)
{
  half2 b13 = __hfma2(__float2half2_rn(2.018f),u1u2,y13);
  half2 b24 = __hfma2(__float2half2_rn(2.018f),u1u2,y24);
  half2 u8_max = __float2half2_rn(255.f);
  half2 u8_min = __float2half2_rn(0.f);
  b13 = __hmax2(__hmin2(u8_max,b13),u8_min);
  b24 = __hmax2(__hmin2(u8_max,b24),u8_min);
  uint32_t ret;
  ((uint8_t*)&ret)[0] = ushort(b13.x);
  ((uint8_t*)&ret)[1] = ushort(b24.x);
  ((uint8_t*)&ret)[2] = ushort(b13.y);
  ((uint8_t*)&ret)[3] = ushort(b24.y);
  return ret;
}

__device__ __forceinline__ uint32_t cvt2g_4x(half2 y13, half2 y24, half2 u1u2,half2 v1v2)
{
  half2 g13 = __hfma2(__float2half2_rn(-0.813f),v1v2,y13);
  half2 g24 = __hfma2(__float2half2_rn(-0.813f),v1v2,y24);
  g13 = __hfma2(__float2half2_rn(-0.391f),u1u2,g13);
  g24 = __hfma2(__float2half2_rn(-0.391f),u1u2,g24);
  half2 u8_max = __float2half2_rn(255.f);
  half2 u8_min = __float2half2_rn(0.f);
  g13 = __hmax2(__hmin2(u8_max,g13),u8_min);
  g24 = __hmax2(__hmin2(u8_max,g24),u8_min);
  uint32_t ret;
  ((uint8_t*)&ret)[0] = ushort(g13.x);
  ((uint8_t*)&ret)[1] = ushort(g24.x);
  ((uint8_t*)&ret)[2] = ushort(g13.y);
  ((uint8_t*)&ret)[3] = ushort(g24.y);
  return ret;
}

template<typename tY>
__device__ __forceinline__ uchar4 cvt2rgb(tY y, tY u, tY v)
{
  float fy =   1.164f  * (max(16,y) - 16);
  float r = fy + 1.596f * ( v - 128);
  float g = fy - 0.813f * ( v - 128) - 0.391f * ( u - 128);
  float b = fy + 2.018f * ( u - 128); 
  uint32_t ir = max(min(255.f,r),0.f);
  uint32_t ig = max(min(255.f,g),0.f);
  uint32_t ib = max(min(255.f,b),0.f);
  uchar4 ret;
  ret.x = ir;
  ret.y = ig;
  ret.z = ib;
  return ret;
}

template<bool IsYU12, bool float_out = false>
__global__ void Yuv2rgb24(uint32_t *__restrict__ in, void *__restrict__ out, int32_t height, int width) {
  uint32_t *u8_out = NULL;
  float2 *fp32_out = NULL;
  if (float_out)
    fp32_out = (float2*)out;
  else
    u8_out = (uint32_t*)out;
  uint32_t w_id = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t h_id = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t out_idx = h_id * width + w_id;
  uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ uint32_t out_sm[768]; // 128 * 4 * 3 / 2


  using dTy = uint16_t; //or int8_t
  if (w_id < width && h_id < height) {
    dTy u1, v1, u2, v2;
    uint32_t temp_y1, temp_y2;
    uchar4 temp_rgb0[4],temp_rgb1[4];
    uint32_t pos = h_id * 2 * width + w_id;
    temp_y1 = in[pos];
    pos  = (h_id * 2 + 1) * width + w_id;
    temp_y2 = in[pos];
    if (IsYU12) {
      pos = height * width * 4 + out_idx;
      uint16_t u12 = ((uint16_t*)in)[pos];
      pos += height * width;
      uint16_t v12 = ((uint16_t*)in)[pos];
      u1 = ((uint8_t*)&u12)[0];
      u2 = ((uint8_t*)&u12)[1];
      v1 = ((uint8_t*)&v12)[0];
      v2 = ((uint8_t*)&v12)[1];
    } else {
      pos = height * width * 2 + out_idx;
      uint32_t uv = in[pos];
      u1 = ((uint8_t*)&uv)[0];
      v1 = ((uint8_t*)&uv)[1];
      u2 = ((uint8_t*)&uv)[2];
      v2 = ((uint8_t*)&uv)[3];
    }
    dTy y1 = ((uint8_t*)&temp_y1)[0];
    dTy y2 = ((uint8_t*)&temp_y1)[1];
    dTy y3 = ((uint8_t*)&temp_y1)[2];
    dTy y4 = ((uint8_t*)&temp_y1)[3];
    dTy y5 = ((uint8_t*)&temp_y2)[0];
    dTy y6 = ((uint8_t*)&temp_y2)[1];
    dTy y7 = ((uint8_t*)&temp_y2)[2];
    dTy y8 = ((uint8_t*)&temp_y2)[3];
    uint8_t *out_row1 = (uint8_t*)out_sm;
    uint8_t *out_row2 = out_row1 + 1536;

    temp_rgb0[0] = cvt2rgb<dTy>(y1,u1,v1);
    temp_rgb0[1] = cvt2rgb<dTy>(y2,u1,v1);
    temp_rgb0[2] = cvt2rgb<dTy>(y3,u2,v2);
    temp_rgb0[3] = cvt2rgb<dTy>(y4,u2,v2);
    temp_rgb1[0] = cvt2rgb<dTy>(y5,u1,v1);
    temp_rgb1[1] = cvt2rgb<dTy>(y6,u1,v1);
    temp_rgb1[2] = cvt2rgb<dTy>(y7,u2,v2); 
    temp_rgb1[3] = cvt2rgb<dTy>(y8,u2,v2); 
    for(int i = 0; i < 4; i++)
    {
      out_row1[thread_idx * 12 + i*3 + 0] = temp_rgb0[i].x;
      out_row1[thread_idx * 12 + i*3 + 1] = temp_rgb0[i].y;
      out_row1[thread_idx * 12 + i*3 + 2] = temp_rgb0[i].z;
      out_row2[thread_idx * 12 + i*3 + 0] = temp_rgb1[i].x;
      out_row2[thread_idx * 12 + i*3 + 1] = temp_rgb1[i].y;
      out_row2[thread_idx * 12 + i*3 + 2] = temp_rgb1[i].z;
    } 
    __syncthreads();

    thread_idx = threadIdx.x;
    int num_loops = float_out ? 6 : 3;
    int out_offset1 = h_id * 2 * width * num_loops + blockDim.x * blockIdx.x * num_loops;
    int out_offset2 = (h_id * 2 + 1) * width * num_loops + blockDim.x * blockIdx.x * num_loops;
    int sm_offset1 = threadIdx.y * blockDim.x * num_loops;
    int sm_offset2 = sm_offset1 + 128 * num_loops;
    int threads = blockDim.x * (blockIdx.x + 1) <= width ? blockDim.x : (width - blockDim.x * blockIdx.x);
    for (int i=0; i<num_loops; i++) {
      if (float_out) {
        uchar2 tmp1, tmp2;
        tmp1 = *((uchar2 *)out_sm + sm_offset1 + thread_idx);
        tmp2 = *((uchar2 *)out_sm + sm_offset2 + thread_idx);
        fp32_out[out_offset1 + thread_idx].x = tmp1.x;
        fp32_out[out_offset1 + thread_idx].y = tmp1.y;

        fp32_out[out_offset2 + thread_idx].x = tmp2.x;
        fp32_out[out_offset2 + thread_idx].y = tmp2.y;
      } else {
        u8_out[out_offset1 + thread_idx] = out_sm[sm_offset1 + thread_idx];
        u8_out[out_offset2 + thread_idx] = out_sm[sm_offset2 + thread_idx];
      }
      thread_idx += threads;
    }
  }
}

template<bool IsYU12>
__global__ void Yuv2rgb24_plane(uint32_t *__restrict__ in, uint32_t *__restrict__ out, int32_t height, int width) {
  uint32_t w_id = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t h_id = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t out_idx = h_id * width + w_id;
  uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
  if (w_id < width && h_id < height) {
    uint8_t u1, v1, u2, v2;
    int16_t u1u2,v1v2;
    uint32_t temp_y1, temp_y2;
    uint32_t pos = h_id * 2 * width + w_id;
    temp_y1 = in[pos];
    pos  = (h_id * 2 + 1) * width + w_id;
    temp_y2 = in[pos];
    if (IsYU12) {
      pos = height * width * 4 + out_idx;
      uint16_t u12 = ((uint16_t*)in)[pos];
      pos += height * width;
      uint16_t v12 = ((uint16_t*)in)[pos];
      u1u2 = __vsub4(u12,0x00008080);
      v1v2 = __vsub4(v12,0x00008080); 
    } else {
      pos = height * width * 2 + out_idx;
      uint32_t uv = in[pos];
      uv = __vsub4(uv,0x80808080);
      u1 = ((uint8_t*)&uv)[0];
      v1 = ((uint8_t*)&uv)[1];
      u2 = ((uint8_t*)&uv)[2];
      v2 = ((uint8_t*)&uv)[3];
      u1u2 = u2 << 8 | u1;
      v1v2 = v2 << 8 | v1;
    }

    temp_y1 = __vmaxu4(0x10101010,temp_y1);
    temp_y1 = __vsub4(temp_y1,0x10101010);
    temp_y2 = __vmaxu4(0x10101010,temp_y2);    
    temp_y2 = __vsub4(temp_y2,0x10101010);

    half2 u12 = short2half2(u1u2);
    half2 v12 = short2half2(v1v2);
    half2 y12 = cvt2half2_low<true>(temp_y1);
    half2 y34 = cvt2half2_high<true>(temp_y1);
    half2 y56 = cvt2half2_low<true>(temp_y2);
    half2 y78 = cvt2half2_high<true>(temp_y2);
    out_idx = h_id * 2 * width + w_id;
    uint32_t r4 = cvt2r_4x(y12,y34,v12);
    uint32_t g4 = cvt2g_4x(y12,y34,u12,v12);
    uint32_t b4 = cvt2b_4x(y12,y34,u12);
    out[out_idx] = r4;
    out[out_idx + width * height * 2] = g4;
    out[out_idx + width * height * 4] = b4;

    out_idx = (h_id * 2 + 1) * width + w_id;
    uint32_t r4_n = cvt2r_4x(y56,y78,v12); 
    uint32_t g4_n = cvt2g_4x(y56,y78,u12,v12);
    uint32_t b4_n = cvt2b_4x(y56,y78,u12);
    out[out_idx] = r4_n;
    out[out_idx + width * height * 2] = g4_n;
    out[out_idx + width * height * 4] = b4_n;
  }
}

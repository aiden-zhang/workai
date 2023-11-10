
#include <cuda_fp16.h>
#include <limits>
using namespace std;

int __device__ pow_2_ceil_log2(int value){

    int current_val=1;
    int i=0;
    do{
       current_val=1<<i;
       i++;
    }while(current_val<value);

    return current_val;
}


__device__ void filter_sort_single_input_kernel(half* in0, int s00, int s01, int s02, int s03, int ss00, int ss01, int ss02, int ss03,
                                            half* in1, int s10, int s11, int s12, int s13, int ss10, int ss11, int ss12, int ss13,
                                            float* out0, int out_s00, int out_s01, int out_ss00, int out_ss01,
                                            int* out1, int out_s10, int out_s11, int out_ss10, int out_ss11,
                                            int* out2, int out_s20, int out_ss20,
                                            float score_threshold, bool IsAscending){

  __shared__ int atomic_idx[2];
  if(threadIdx.x==0){
    atomic_idx[0]=0;
  }
  __syncthreads();

  int batch_idx=blockIdx.x;
  int c=s11/s01;
  for(int tidx=threadIdx.x;tidx<s01*s12*s13;tidx+=blockDim.x){

      int pre_idx=tidx/s13;
      int dim3_idx=tidx%s13;
      int dim2_idx=pre_idx%s12;
      int dim1_idx=pre_idx/s12;

      float value = in0[(batch_idx/c)*ss00+dim1_idx*ss01+dim2_idx*ss02+dim3_idx];
      value *= __half2float(in1[(batch_idx/c)*ss10+(batch_idx%c*s01+dim1_idx)*ss11+dim2_idx*ss12+dim3_idx]);

      if(value>=score_threshold){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = value;
        out1[batch_idx*out_ss10+idx] = (dim2_idx*s13+dim3_idx)*s01+dim1_idx;

      }
  }

  __syncthreads();

  if(threadIdx.x==0){
    out2[batch_idx] = atomic_idx[0];
    atomic_idx[1]=pow_2_ceil_log2(atomic_idx[0]);
  }
  __syncthreads();
  float float_min = std::numeric_limits<float>::min();

  for(int tidx=atomic_idx[0];tidx<atomic_idx[1];tidx+=blockDim.x){
    out0[batch_idx*out_ss00+tidx] = float_min;
  }
  __syncthreads();
  int sort_size=atomic_idx[1];


  for(int stride=1;stride<=(sort_size>>1);stride=stride*2){
    for(int stride_=stride;stride_>=1;stride_=stride_>>1){
      
      for(int tidx=threadIdx.x;tidx<(sort_size>>1);tidx+=blockDim.x){
        int sort_block_id=tidx/stride_;
        int idx=sort_block_id*(stride_<<1);
        idx=idx+tidx%(stride_);
        
        auto ele0 = out0[batch_idx*out_ss00+idx];
        auto ele0_idx = out1[batch_idx*out_ss10+idx];

        // todo check
        auto ele1 = out0[batch_idx*out_ss00+idx+stride_];
        auto ele1_idx = out1[batch_idx*out_ss10+idx+stride_];

        if(IsAscending){
          if(ele0<ele1){

          }
          else{
            auto tmp_ele = ele0;
            auto tmp_ele_idx = ele0_idx;
            ele0 = ele1;
            ele0_idx = ele1_idx;
            ele1 = tmp_ele;
            ele1_idx = tmp_ele_idx;            
          }
        }
        else{
          if(ele0<ele1){
            auto tmp_ele = ele0;
            auto tmp_ele_idx = ele0_idx;
            ele0 = ele1;
            ele0_idx = ele1_idx;
            ele1 = tmp_ele;
            ele1_idx = tmp_ele_idx;            
          }
          else{

          }
        }

        if((idx/(stride<<1)) %2){
            auto tmp_ele=ele0;
            auto tmp_ele_idx=ele0_idx;
              ele0=ele1;
              ele0_idx=ele1_idx;
              ele1=tmp_ele;
              ele1_idx=tmp_ele_idx;
        }

        out0[batch_idx*out_ss00+idx] = ele0;
        out1[batch_idx*out_ss10+idx] = ele0_idx;
        out0[batch_idx*out_ss00+idx+stride_] = ele1;
        out1[batch_idx*out_ss10+idx+stride_] = ele1_idx;

      }
    }
  }  

}

using namespace std;
auto __device__ half2mul(float a,float b){

    float a_v=a;
    float b_v=b;

    half2 a_half2=*(half2*)(&a_v);
    half2 b_half2=*(half2*)(&b_v);
    half2 c;

    c.x=a_half2.x*b_half2.x;
    c.y=a_half2.y*b_half2.y;
    return c;
}
void __device__ half8mul(float4 a,float4 b,half* half8){
    half2 half2_value=half2mul(a.x,b.x);

    half8[0]=half2_value.x;
    half8[1]=half2_value.y;

    half2_value=half2mul(a.y,b.y);
    half8[2]=half2_value.x;
    half8[3]=half2_value.y;

    half2_value=half2mul(a.z,b.z);
    half8[4]=half2_value.x;
    half8[5]=half2_value.y;

    half2_value=half2mul(a.w,b.w);
    half8[6]=half2_value.x;
    half8[7]=half2_value.y;

}

void __device__ filter_sort_single_input_half8_kernel(void* in0, int s00, int s01, int s02, int s03, int ss00, int ss01, int ss02, int ss03,
                                                  void* in1, int s10, int s11, int s12, int s13, int ss10, int ss11, int ss12, int ss13,
                                                  float* out0, int out_s00, int out_s01, int out_ss00, int out_ss01,
                                                  int* out1, int out_s10, int out_s11, int out_ss10, int out_ss11,
                                                  int* out2, int out_s20, int out_ss20,
                                                  float score_threshold, bool IsAscending){

  float4* f4in0 = (float4*)in0;
  float4* f4in1 = (float4*)in1;  
  
  ss00/=8;
  ss01/=8;
  ss02/=8;
  ss10/=8;
  ss11/=8;
  ss12/=8;

  __shared__ int atomic_idx[2];
  if(threadIdx.x==0){
    atomic_idx[0]=0;
  }
  __syncthreads();

  int batch_idx=blockIdx.x;
  half half_thresh = __float2half(score_threshold);

  int c=s11/s01;
  for(int tidx=threadIdx.x;tidx<s01*s12*s13/8;tidx+=blockDim.x){
      int pre_idx=tidx/(s13/8);
      int dim3_idx=tidx%(s13/8);
      int dim2_idx=pre_idx%s12;
      int dim1_idx=pre_idx/s12;

      auto value0 = f4in0[(batch_idx/c)*ss00+dim1_idx*ss01+dim2_idx*ss02+dim3_idx];
      auto value1 = f4in1[(batch_idx/c)*ss10+(batch_idx%c*s01+dim1_idx)*ss11+dim2_idx*ss12+dim3_idx];
      half half8[8];

      half8mul(value0,value1,half8);
      int address=((dim2_idx*s13+dim3_idx*8)*s01+dim1_idx);

      if(half8[0]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[0];
        out1[batch_idx*out_ss10+idx] = address;
      }
      if(half8[1]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[1];
        out1[batch_idx*out_ss10+idx] = address+s01;
      }
      if(half8[2]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[2];
        out1[batch_idx*out_ss10+idx] = address+s01*2;
      }
      if(half8[3]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[3];
        out1[batch_idx*out_ss10+idx] = address+s01*3;
      }
      if(half8[4]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[4];
        out1[batch_idx*out_ss10+idx] = address+s01*4;
      }
      if(half8[5]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[5];
        out1[batch_idx*out_ss10+idx] = address+s01*5;
      }
      if(half8[6]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[6];
        out1[batch_idx*out_ss10+idx] = address+s01*6;
      }
      if(half8[7]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[7];
        out1[batch_idx*out_ss10+idx] = address+s01*7;
      }
  }


  __syncthreads();

  if(threadIdx.x==0){
    out2[batch_idx] = atomic_idx[0];
    atomic_idx[1]=pow_2_ceil_log2(atomic_idx[0]);
  }
  __syncthreads();

  float float_min = std::numeric_limits<float>::min();

  for(int tidx=atomic_idx[0];tidx<atomic_idx[1];tidx+=blockDim.x){
    out0[batch_idx*out_ss00+tidx] = float_min;
  }
  __syncthreads();
  
  int sort_size=atomic_idx[1];
  for(int stride=1;stride<=(sort_size>>1);stride=stride*2){
    for(int stride_=stride;stride_>=1;stride_=stride_>>1){
      
      for(int tidx=threadIdx.x;tidx<(sort_size>>1);tidx+=blockDim.x){
        int sort_block_id=tidx/stride_;
        int idx=sort_block_id*(stride_<<1);
        idx=idx+tidx%(stride_);
        
        auto ele0 = out0[batch_idx*out_ss00+idx];
        auto ele0_idx = out1[batch_idx*out_ss10+idx];

        // todo check
        auto ele1 = out0[batch_idx*out_ss00+idx+stride_];
        auto ele1_idx = out1[batch_idx*out_ss10+idx+stride_];

        if(IsAscending){
          if(ele0<ele1){

          }
          else{
            auto tmp_ele = ele0;
            auto tmp_ele_idx = ele0_idx;
            ele0 = ele1;
            ele0_idx = ele1_idx;
            ele1 = tmp_ele;
            ele1_idx = tmp_ele_idx;            
          }
        }
        else{
          if(ele0<ele1){
            auto tmp_ele = ele0;
            auto tmp_ele_idx = ele0_idx;
            ele0 = ele1;
            ele0_idx = ele1_idx;
            ele1 = tmp_ele;
            ele1_idx = tmp_ele_idx;            
          }
          else{

          }
        }

        if((idx/(stride<<1)) %2){
            auto tmp_ele=ele0;
            auto tmp_ele_idx=ele0_idx;
              ele0=ele1;
              ele0_idx=ele1_idx;
              ele1=tmp_ele;
              ele1_idx=tmp_ele_idx;
        }

        out0[batch_idx*out_ss00+idx] = ele0;
        out1[batch_idx*out_ss10+idx] = ele0_idx;
        out0[batch_idx*out_ss00+idx+stride_] = ele1;
        out1[batch_idx*out_ss10+idx+stride_] = ele1_idx;

      }
    }
  }  
}

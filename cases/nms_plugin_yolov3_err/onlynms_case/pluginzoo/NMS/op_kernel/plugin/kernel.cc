#include "kernel.h"
    char combine_src_file[] = R"code(
      template<typename BoxType,typename ScoreType,typename SelectType,
            typename CsumType,typename IdxType>
__device__ void CombineNonMaxSuppressionPost_device_func(
                                        BoxType* boxes_buffer,
                                        int b_s0,int b_s1,int b_s2,
                                        int b_ss0,int b_ss1,int b_ss2,

										ScoreType* scores_buffer,
										int s_s0,int s_s1,int s_s2,
                                        int s_ss0,int s_ss1,int s_ss2,

										SelectType* selected_ids_buffer,
										int sel_s0,int sel_s1,int sel_s2,
                                        int sel_ss0,int sel_ss1,int sel_ss2,

										CsumType* csum_buffer,
										int csum_s0,int csum_ss0,

										IdxType* out_idxs_buffer,
										int out_idx_s0,int out_idx_s1,
										int out_idx_ss0,int out_idx_ss1,

										BoxType* out_boxes_buffer,
										int out_box_s0,
										int out_box_ss0,

										ScoreType* out_scores_buffer,
										int out_score_s0,
										int out_score_ss0,

                                        SelectType* valid_num=nullptr
										){



    int B=b_s0;
    int C=b_s1;
    int boxes_c=C;

    int count=csum_buffer[B*C-1];

	int tidx=threadIdx.x+blockIdx.x*blockDim.x;
    if(tidx==0){
        valid_num[0]=count;
    }

    int start=-1;
    int end=B*C-1;

    int current_b=0;
    int current_c=0;
    int idx=0;
    if(tidx<count){

        while(true){
            int current_idx=(start+1+end)>>1;

            int value=current_idx-1>=0?csum_buffer[current_idx-1]:-1;


            if(value <= tidx && tidx<csum_buffer[current_idx]){
                current_c=(current_idx)%C;
                current_b=(current_idx)/C;
                idx=(value>0 ? tidx-value:tidx);
                break;
            }else if(tidx>=csum_buffer[current_idx]){
                start=current_idx;
            }else if(tidx<value){
                end=current_idx;
            }
        }

        auto idx_tmp=selected_ids_buffer[current_b*sel_ss0+current_c*sel_ss1+idx*sel_ss2];

            int boxes_c_idx=(boxes_c==1?0:current_c);

            auto box_value=((float4*)boxes_buffer)[current_b*b_ss0+boxes_c_idx*b_ss1+idx_tmp*b_ss2];
            ((float4*)out_boxes_buffer)[tidx*out_box_ss0]=box_value;

            auto score_value=scores_buffer[current_b*s_ss0+current_c*s_ss1+idx_tmp*s_ss2];
            out_scores_buffer[tidx]=score_value;

            out_idxs_buffer[tidx*out_idx_ss0+0]=current_b;
            out_idxs_buffer[tidx*out_idx_ss0+1*out_idx_ss1]=current_c;

    }

}


extern "C" __global__ void fused_dl_combine_non_max_suppression_post_kernel(
    float* __restrict__ placeholder,int s0_s0,int s0_s1,int s0_s2,int s0_ss0,int s0_ss1,int s0_ss2,
     float* __restrict__ placeholder1,int s1_s0,int s1_s1,int s1_s2,int s1_ss0,int s1_ss1,int s1_ss2,
      int* __restrict__ placeholder2,int s2_s0,int s2_s1,int s2_s2,int s2_ss0,int s2_ss1,int s2_ss2,
      int* __restrict__ placeholder4,int s3_s0,int s3_ss0,

       int* __restrict__ combine_non_max_suppression_post_op_v0,int o0_s0,int o0_s1,int o0_ss0,int o0_ss1,
       float* __restrict__ combine_non_max_suppression_post_op_v1,int o1_s0,int o1_s1,int o1_ss0,int o1_ss1,
        float* __restrict__ combine_non_max_suppression_post_op_v2,int o2_s0,int o2_ss0,
         int* __restrict__ combine_non_max_suppression_post_op_v3) {

   CombineNonMaxSuppressionPost_device_func<float,float,int,int,int>
  (placeholder,s0_s0,s0_s1,s0_s2,s0_ss0,s0_ss1,s0_ss2,
   placeholder1,s1_s0,s1_s1,s1_s2,s1_ss0,s1_ss1,s1_ss2,
   placeholder2,s2_s0,s2_s1,s2_s2,s2_ss0,s2_ss1,s2_ss2,

   placeholder4,s3_s0,s3_ss0,
   combine_non_max_suppression_post_op_v0, o0_s0,o0_s1,o0_ss0,o0_ss1,
   combine_non_max_suppression_post_op_v1, o1_s0,o1_ss0,
   combine_non_max_suppression_post_op_v2, o2_s0,o2_ss0,
   combine_non_max_suppression_post_op_v3);
}
    )code";

char cusm_src_file[] = R"code(
    namespace{
template<int value>
__device__ constexpr int Log2(){
	for(int i=31;i>=0;i--){
	 	if (1==(value>>i)){
	 		return i;
	 	}
	}
	return -1;
}

template<typename T,int WAY_NUM,int log2_wrap_size=5>
__device__ inline void CSum_wrap_inner(T *value) {
        constexpr int wrap_size=1<<log2_wrap_size;

        int LaneId=threadIdx.x&(wrap_size-1);
        for (int stride = 1; stride <= (wrap_size>>1); stride = stride << 1) {
            T target_value[WAY_NUM];
             for (int j = 0; j < WAY_NUM; j++) {
                        target_value[j]=__shfl_up_sync(0xffffffff,value[j],stride,wrap_size);
             }
             if((LaneId & (2*(stride-1)+1)) == 2*(stride-1)+1){
                   for (int j = 0; j < WAY_NUM; j++) {
                                value[j]+=target_value[j];
                   }
             }
        }
        int i=1;
        for (int stride = (wrap_size >> 1); stride >= 2; stride = stride >> 1) {
            T target_value[WAY_NUM];
            for(int j=0;j<WAY_NUM;j++){
               target_value[j]=__shfl_up_sync(0xffffffff,value[j],stride>>1,wrap_size);
            }
            if(LaneId >=stride && LaneId == stride*(LaneId>>(log2_wrap_size-i)) -1+(stride>>1)){
                for(int j=0;j<WAY_NUM;j++)
                   value[j]+=target_value[j];

            }
            i++;
        }
}

template<typename T,int WAY_NUM,int LOG2_BLOCK_SIZE,int log2_wrap_size=5>
__device__ inline void CSumSync(T* ele){
	CSum_wrap_inner<T,WAY_NUM,log2_wrap_size>(ele);

	if(LOG2_BLOCK_SIZE>log2_wrap_size){
			constexpr int num=1<<(LOG2_BLOCK_SIZE-log2_wrap_size);
			__shared__ T value[num*WAY_NUM];
			if((threadIdx.x & ((1<<log2_wrap_size)-1))==((1<<log2_wrap_size)-1)){
				for(int i=0;i<WAY_NUM;i++){
					value[(threadIdx.x>>log2_wrap_size)*WAY_NUM+i]=ele[i];
				}
			}

			__syncthreads();

			T ele_sum[WAY_NUM];
			if(threadIdx.x<num){
				for(int i=0;i<WAY_NUM;i++){
					ele_sum[i]=value[threadIdx.x*WAY_NUM+i];
				}
			}
			CSum_wrap_inner<T,WAY_NUM,Log2<num>()>(ele_sum);


			if(threadIdx.x<num)
				for(int i=0;i<WAY_NUM;i++)
					value[threadIdx.x*WAY_NUM+i]=ele_sum[i];

			__syncthreads();

			if((threadIdx.x>>5)>=1)
				for(int i=0;i<WAY_NUM;i++)
					ele[i]=ele[i]+value[((threadIdx.x>>5)-1)*WAY_NUM+i];
	}
}


template<typename DATA_TYPE,int BLOCK_SIZE=256>
__device__ void csum_device_func(DATA_TYPE* in,int B,int N,int s0,int s1,
	DATA_TYPE* out,int os0,int os1){

	 __shared__  DATA_TYPE last_sum[1];
	 if(threadIdx.x==BLOCK_SIZE-1){
	 	last_sum[0]=0;
	 }
	 for(int i=0;i<(N+BLOCK_SIZE-1)/BLOCK_SIZE;i++){

		int batch_idx=blockIdx.x;
		int e_idx=threadIdx.x+i*BLOCK_SIZE;
        auto ele_value = e_idx < N ? in[batch_idx*s0+e_idx*s1]:DATA_TYPE(0);

        CSumSync<DATA_TYPE,1,(Log2<BLOCK_SIZE>()),5>(&ele_value);

		if(threadIdx.x+i*BLOCK_SIZE<N){
		 	out[batch_idx*os0+e_idx*os1]=ele_value+last_sum[0];
		}
		 __syncthreads();

		 if(threadIdx.x==BLOCK_SIZE-1){
		 	last_sum[0]+=ele_value;
		 }
	 }
}
}

extern "C" __global__ void fused_dl_csum_kernel0(int* __restrict__ placeholder,int s0,int s1,int ss0,int ss1,
        int* __restrict__ dl_csum_op,int os0,int os1,int oss0,int oss1) {
  csum_device_func<int,256>(placeholder, s0, s1,ss0,ss1, dl_csum_op,oss0,oss1);
}

)code";

char filter_sort_src_file[] = R"code(
    
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


__device__ void fused_dl_filter_sort_kernel(half* in0, int s00, int s01, int s02, int s03, int ss00, int ss01, int ss02, int ss03,
                                            half* in1, int s10, int s11, int s12, int s13, int ss10, int ss11, int ss12, int ss13,
                                            half* in2, int s20, int s21, int s22, int s23, int ss20, int ss21, int ss22, int ss23,
                                            half* in3, int s30, int s31, int s32, int s33, int ss30, int ss31, int ss32, int ss33,
                                            half* in4, int s40, int s41, int s42, int s43, int ss40, int ss41, int ss42, int ss43,
                                            half* in5, int s50, int s51, int s52, int s53, int ss50, int ss51, int ss52, int ss53,
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

  int offset=s01*s12*s13;

  c=s31/s21;
  for(int tidx=threadIdx.x; tidx<(s21*s32*s33); tidx += blockDim.x)
  {   
      int pre_idx=tidx/s33;
      int dim3_idx=tidx%s33;
      int dim2_idx=pre_idx%s32;
      int dim1_idx=pre_idx/s32;

      float value = in2[(batch_idx/c)*ss20+dim1_idx*ss21 + dim2_idx*ss22 + dim3_idx];
      value *= __half2float(in3[(batch_idx/c)*ss30+(batch_idx%c*s21+dim1_idx)*ss31+dim2_idx*ss32+dim3_idx]);
      if(value>=score_threshold){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = value;
        out1[batch_idx*out_ss10+idx] = offset+(dim2_idx*s33+dim3_idx)*s21+dim1_idx;

      }
  }

  offset+=s21*s32*s33;

  c=s51/s41;
  for(int tidx=threadIdx.x;tidx<s41*s52*s53;tidx+=blockDim.x){
      int pre_idx=tidx/s53;
      int dim3_idx=tidx%s53;
      int dim2_idx=pre_idx%s52;
      int dim1_idx=pre_idx/s52;

      float value = in4[(batch_idx/c)*ss40+dim1_idx*ss41+dim2_idx*ss42+dim3_idx];
      value *= __half2float(in5[(batch_idx/c)*ss50+(batch_idx%c*s41+dim1_idx)*ss51+dim2_idx*ss52+dim3_idx]);
      if(value>=score_threshold){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = value;
        out1[batch_idx*out_ss10+idx] = offset+(dim2_idx*s53+dim3_idx)*s41+dim1_idx;

      }
  }

  __syncthreads();

  if(threadIdx.x==0){
    out2[batch_idx] = atomic_idx[0];
    atomic_idx[1]=pow_2_ceil_log2(atomic_idx[0]);
  }
  __syncthreads();
  float float_min = std::numeric_limits<float>::min();

  for(int tidx=atomic_idx[0]+threadIdx.x;tidx<atomic_idx[1];tidx+=blockDim.x){
    out0[batch_idx*out_ss00+tidx] = float_min;
  }
  __syncthreads();
  int sort_size=atomic_idx[1];


  for(int stride=1;stride<=(sort_size>>1);stride=stride*2){
    for(int stride_=stride;stride_>=1;stride_=stride_>>1){
      __syncthreads();
      __threadfence();
      
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

void __device__ fused_dl_filter_sort_half8_kernel(void* in0, int s00, int s01, int s02, int s03, int ss00, int ss01, int ss02, int ss03,
                                                  void* in1, int s10, int s11, int s12, int s13, int ss10, int ss11, int ss12, int ss13,
                                                  void* in2, int s20, int s21, int s22, int s23, int ss20, int ss21, int ss22, int ss23,
                                                  void* in3, int s30, int s31, int s32, int s33, int ss30, int ss31, int ss32, int ss33,
                                                  void* in4, int s40, int s41, int s42, int s43, int ss40, int ss41, int ss42, int ss43,
                                                  void* in5, int s50, int s51, int s52, int s53, int ss50, int ss51, int ss52, int ss53,
                                                  float* out0, int out_s00, int out_s01, int out_ss00, int out_ss01,
                                                  int* out1, int out_s10, int out_s11, int out_ss10, int out_ss11,
                                                  int* out2, int out_s20, int out_ss20,
                                                  float score_threshold, bool IsAscending){

  float4* f4in0 = (float4*)in0;
  float4* f4in1 = (float4*)in1;  
  float4* f4in2 = (float4*)in2;  
  float4* f4in3 = (float4*)in3;  
  float4* f4in4 = (float4*)in4;  
  float4* f4in5 = (float4*)in5;  
  
  ss00/=8;
  ss01/=8;
  ss02/=8;
  ss10/=8;
  ss11/=8;
  ss12/=8;
  ss20/=8;
  ss21/=8;
  ss22/=8;
  ss30/=8;
  ss31/=8;
  ss32/=8;
  ss40/=8;
  ss41/=8;
  ss42/=8;
  ss50/=8;
  ss51/=8;
  ss52/=8;

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

  int offset=s01*s12*s13;

  c=s31/s21;
  for(int tidx=threadIdx.x;tidx<s21*s32*(s33/8);tidx+=blockDim.x){
      int pre_idx=tidx/(s33/8);
      int dim3_idx=tidx%(s33/8);
      int dim2_idx=pre_idx%s32;
      int dim1_idx=pre_idx/s32;

      auto value0 = f4in2[(batch_idx/c)*ss20+dim1_idx*ss21 + dim2_idx*ss22 + dim3_idx];
      auto value1 = f4in3[(batch_idx/c)*ss30+(batch_idx%c*s21+dim1_idx)*ss31+dim2_idx*ss32+dim3_idx];
      half half8[8];

      half8mul(value0,value1,half8);
      int address=((dim2_idx*s13+dim3_idx*8)*s01+dim1_idx);

      if(half8[0]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[0];
        out1[batch_idx*out_ss10+idx] = offset+address;
      }
      if(half8[1]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[1];
        out1[batch_idx*out_ss10+idx] = offset+address+s01;
      }
      if(half8[2]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[2];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*2;
      }
      if(half8[3]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[3];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*3;
      }
      if(half8[4]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[4];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*4;
      }
      if(half8[5]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[5];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*5;
      }
      if(half8[6]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[6];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*6;
      }
      if(half8[7]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[7];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*7;
      }
  }

  offset+=s21*s32*s33;

  c=s51/s41;
  for(int tidx=threadIdx.x;tidx<s41*s52*s53/8;tidx+=blockDim.x){
      int pre_idx=tidx/(s53/8);
      int dim3_idx=tidx%(s53/8);
      int dim2_idx=pre_idx%s52;
      int dim1_idx=pre_idx/s52;

      auto value0 = f4in4[(batch_idx/c)*ss40+dim1_idx*ss41+dim2_idx*ss42+dim3_idx];
      auto value1 = f4in5[(batch_idx/c)*ss50+(batch_idx%c*s41+dim1_idx)*ss51+dim2_idx*ss52+dim3_idx];
      half half8[8];

      half8mul(value0,value1,half8);
      int address=((dim2_idx*s13+dim3_idx*8)*s01+dim1_idx);

      if(half8[0]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[0];
        out1[batch_idx*out_ss10+idx] = offset+address;
      }
      if(half8[1]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[1];
        out1[batch_idx*out_ss10+idx] = offset+address+s01;
      }
      if(half8[2]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[2];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*2;
      }
      if(half8[3]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[3];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*3;
      }
      if(half8[4]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[4];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*4;
      }
      if(half8[5]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[5];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*5;
      }
      if(half8[6]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[6];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*6;
      }
      if(half8[7]>=half_thresh){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = half8[7];
        out1[batch_idx*out_ss10+idx] = offset+address+s01*7;
      }
  }

  __syncthreads();

  if(threadIdx.x==0){
    out2[batch_idx] = atomic_idx[0];
    atomic_idx[1]=pow_2_ceil_log2(atomic_idx[0]);
  }
  __syncthreads();

  float float_min = std::numeric_limits<float>::min();

  for(int tidx=atomic_idx[0]+threadIdx.x;tidx<atomic_idx[1];tidx+=blockDim.x){
    out0[batch_idx*out_ss00+tidx] = float_min;
  }
  __syncthreads();
  
  int sort_size=atomic_idx[1];
  for(int stride=1;stride<=(sort_size>>1);stride=stride*2){
    for(int stride_=stride;stride_>=1;stride_=stride_>>1){
      __syncthreads();
      __threadfence();
      
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

__device__ void fused_dl_filter_sort_pure_kernel(float* in0, int s00, int s01, int ss00, int ss01,
                                            float* out0, int out_s00, int out_s01, int out_ss00, int out_ss01,
                                            int* out1, int out_s10, int out_s11, int out_ss10, int out_ss11,
                                            int* out2, int out_s20, int out_ss20,
                                            float score_threshold, bool IsAscending){
  /*
  block size: 128
  block num: class_num*bacth_size
  s00: bacth_size*class_num
  s01: num anchor
  */

  __shared__ int atomic_idx[2];
  if(threadIdx.x==0){
    atomic_idx[0]=0;
  }
  __syncthreads();

  int batch_idx=blockIdx.x;
  
  for(int tidx=threadIdx.x; tidx<s01; tidx+=blockDim.x){
      int dim1_idx = tidx%s01;
      float value = in0[batch_idx*ss00 + dim1_idx];

      if(value>=score_threshold){
        int idx=atomicAdd(&atomic_idx[0],1);
        out0[batch_idx*out_ss00+idx] = value;
        out1[batch_idx*out_ss00+idx] = dim1_idx;

      }
  }

  __syncthreads();

  if(threadIdx.x==0){
    out2[batch_idx] = atomic_idx[0];
    atomic_idx[1]=pow_2_ceil_log2(atomic_idx[0]);
  }
  __syncthreads();
  float float_min = std::numeric_limits<float>::min();

  for(int tidx=atomic_idx[0]+threadIdx.x; tidx<atomic_idx[1]; tidx+=blockDim.x){
    out0[batch_idx*out_ss00+tidx] = float_min;
  }
  __syncthreads();
  int sort_size=atomic_idx[1];


  for(int stride=1;stride<=(sort_size>>1);stride=stride*2){
    for(int stride_=stride;stride_>=1;stride_=stride_>>1){
      __syncthreads();
      __threadfence();
      
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

)code";

char gather_boxes_src_file[] = R"code(
    #include <cuda_runtime.h>

__device__  void gather_boxes_device_func(
															const float4* boxes_buffer,
                              const int* ids_buffer,
                              float4* out_boxes_buffer,
                              const int q,
                              const int B, const int C, const int N,
                              const int ids_len,
                              const int* sort_size_buffer=nullptr)
{
	if(q==1)
	{
    int c_idx=blockIdx.x%C;
    int b_idx=blockIdx.x/C;

    int box_dim_sizes[2]={B,N};
    int idx_dim_sizes[2]={B*C,ids_len};
    int sort_dim_sizes[1]={B*C};

    if(sort_size_buffer!=nullptr)
    {
  		for(int i=threadIdx.x;i<sort_size_buffer[(b_idx*C+c_idx)];i=i+blockDim.x)
        {          
        		auto idx=ids_buffer[((b_idx*C+c_idx)*ids_len + i)];

            if(idx>=0 && idx< N)
            {
            		auto box=boxes_buffer[(b_idx*N + idx)];
                out_boxes_buffer[((b_idx*C+c_idx)*N + i)]=box;
            }
        }
    }
    else
    {
        for(int i=threadIdx.x; i<N; i+=blockDim.x){

            
        		auto idx=ids_buffer[((b_idx*C+c_idx)*ids_len + i)];

            
            auto box=boxes_buffer[(b_idx*N + idx)];

           
            out_boxes_buffer[((b_idx*C+c_idx)*N + i)] = box;
        }
    }

  }
  else
  {
        int c_idx=blockIdx.x%C;
        int b_idx=blockIdx.x/C;
        int box_dim_sizes[2]={B*C,N};
        int idx_dim_sizes[2]={B*C,ids_len};
        int sort_dim_sizes[1]={B*C};


        if(sort_size_buffer!=nullptr)
        {
            
        		for(int i=threadIdx.x; i<sort_size_buffer[(b_idx*C+c_idx)]; i=i+blockDim.x)
            {
                
            		auto idx = ids_buffer[((b_idx*C+c_idx)*ids_len + i)];

                if(idx>=0 && idx< N)
                {
                    
                		auto box = boxes_buffer[((b_idx*C+c_idx)*N + idx)];
                    
                    out_boxes_buffer[((b_idx*C+c_idx)*N + i)] = box;

                }
            }
        }
        else
        {
            
        		for(int i=threadIdx.x; i<N; i+=blockDim.x)
            {

                
            		auto idx = ids_buffer[((b_idx*C+c_idx)*ids_len + i)];

               
                auto box = boxes_buffer[((b_idx*C+c_idx)*N + idx)];

                
                out_boxes_buffer[((b_idx*C+c_idx)*N + i)] = box;
            }
        }

  }
}


extern "C" __global__  void gather_boxes_device_func_global(
															const float4* boxes_buffer,
                              const int* ids_buffer,
                              float4* out_boxes_buffer,
                              const int q,
                              const int B, const int C, const int N,
                              const int ids_len,
                              const int* sort_size_buffer=nullptr)
{

	gather_boxes_device_func(boxes_buffer,
                              ids_buffer,
                              out_boxes_buffer,
                              q,
                              B, C, N,
                              ids_len,
                              sort_size_buffer);

}

__device__  void gather_boxes_device_func_kernel(
                              const float4* boxes_buffer,
                              const int* ids_buffer,
                              float4* out_boxes_buffer,
                              const int q,
                              const int B, const int C, const int N,
                              const int ids_len,
                              const int* sort_size_buffer=nullptr)
{

  gather_boxes_device_func(boxes_buffer,
                              ids_buffer,
                              out_boxes_buffer,
                              q,
                              B, C, N,
                              ids_len,
                              sort_size_buffer);

}
)code";

char non_max_suppression_src_file[] = R"code(
    #include <cuda_runtime.h>


#define MAX(a,b) ((a)>(b) ? (a):(b))
#define MIN(a,b) ((a)<(b) ? (a):(b))

template<typename T>
struct BaseTypeFromVector
{
  typedef T type;
};
template<>
struct BaseTypeFromVector<float4>
{
  typedef float type;
};

template<typename T,int center_point_box>
class Box{
public:
  using BaseT =typename BaseTypeFromVector<T>::type;
  __device__  Box(T a)
  {
    this->y0=MIN(a.x,a.z);
    this->x0=MIN(a.y,a.w);
    this->y1=MAX(a.x,a.z);
    this->x1=MAX(a.y,a.w);
  }

  __device__ BaseT area(){
    return (y1-y0)*(x1-x0);

  }
  BaseT y0;
  BaseT x0;
  BaseT y1;
  BaseT x1;
};

template<typename T>
class Box<T,1>
{
public:
  using BaseT =typename BaseTypeFromVector<T>::type;

  __device__  Box(T a){
    BaseT x_center=a.x;
    BaseT y_center=a.y;
    this->width=a.z;
    this->height=a.w;

    this->y0=y_center-height/2;
    this->x0=x_center-width/2;
    this->y1=y0+height;
    this->x1=x0+width;

  }

  __device__ BaseT area(){
    return height*width;
  }
  BaseT y0;
  BaseT x0;
  BaseT y1;
  BaseT x1;
  BaseT height;
  BaseT width;
};

template<typename BOX,typename T>
__device__ bool IOU_GE(BOX a,BOX b,T iou_threshold)
{
  // T y_diff=MAX(0,MIN(a.y1,b.y1)-MAX(a.y0,b.y0));
  // T x_diff=MAX(0,MIN(a.x1,b.x1)-MAX(a.x0,b.x0));

  // return y_diff*x_diff>iou_threshold*(a.area()+b.area()-y_diff*x_diff);
  T ymin_i = MIN(a.y0,a.y1);
  T xmin_i = MIN(a.x0,a.x1);
  T ymax_i = MAX(a.y0,a.y1);
  T xmax_i = MAX(a.x0,a.x1);

  T ymin_j = MIN(b.y0,b.y1);
  T xmin_j = MIN(b.x0,b.x1);
  T ymax_j = MAX(b.y0,b.y1);
  T xmax_j = MAX(b.x0,b.x1);

  T area_i = (ymax_i - ymin_i)*(xmax_i - xmin_i);
  T area_j = (ymax_j - ymin_j)*(xmax_j - xmin_j);

  if(area_i <=0 || area_j <= 0) return false;

  T interection_ymin = MAX(ymin_i,ymin_j);
  T interection_xmin = MAX(xmin_i,xmin_j);
  T interection_ymax = MIN(ymax_i,ymax_j);
  T interection_xmax = MIN(xmax_i,xmax_j);

  T interection_area = MAX((interection_ymax-interection_ymin),0)*
                        MAX((interection_xmax-interection_xmin),0);

  return interection_area>iou_threshold*(area_i+area_j-interection_area);
}

template<typename OP0,
         typename OP1,
         typename OP2,
         typename OP3,
         typename OP4,
         typename OP5,
         typename TensorBool,
         typename TensorIdx,
         typename TensorCount,
         int center_point_box=1>
class NonMaxSuppressionOp
{
public:
  using BOX = Box<float4, center_point_box>;

  __device__ NonMaxSuppressionOp(OP0 boxes, //tensor ptr
                                 OP1 scores, //tensor ptr
                                 OP2 max_output_boxes_per_class, //scalar
                                 OP3 iou_threshold, //scalar
                                 OP4 score_threshold, //scalar
                                 OP5 sort_size, //scalar
                                 TensorBool disable, //tensor ptr
                                 TensorIdx output_ids, //tensor ptr
                                 TensorCount count, //ptr one num 
                                 int* socres_dim_sizes
                                 )
  {
    this->boxes=boxes;
    this->scores=scores;
    this->max_output_boxes_per_class=max_output_boxes_per_class;
    this->iou_threshold=iou_threshold;
    this->score_threshold=score_threshold;
    this->sort_size=sort_size;

    this->disable=disable;
    this->output_ids=output_ids;
    this->count=count;

    __shared__ float4 current_box[1];
    __shared__ bool is_finshed[1];
    __shared__ int stride[2];
    
    this->current_box=current_box;
    this->is_finshed=is_finshed;

    this->stride=stride;

    this->socres_dim_sizes = socres_dim_sizes;
  }

  template<typename IndexHelper> 
  __device__ void AutoLoad(IndexHelper idx)
  {
    
    
    auto iou_threshold = this->iou_threshold;
    float score_threshold;
    
    int max_output_boxes;

    int count=0;
    if(threadIdx.x==0)
    {
      
      score_threshold = (float)(this->score_threshold);
      
      max_output_boxes = this->max_output_boxes_per_class;

      this->stride[0] = 0;
      if(max_output_boxes == 0)
      {
        max_output_boxes=-1;
      }
      int start_idx = 0;
      
      int end_idx = socres_dim_sizes[0] - 1;
      
      if(sort_size >= 0)
      {
        end_idx = sort_size - 1;
      }
      while(start_idx+1 >= end_idx && start_idx != end_idx)
      {
              int center_idx = (start_idx + end_idx + 1)>>1;
              
              auto current_score = scores[center_idx];
              
              if(current_score < score_threshold)
              {
                  end_idx=center_idx;
              }
              else
              {
                  start_idx=center_idx;
              }
       }
       this->stride[1] = end_idx;//find last score
    }
    

    
    int iter_idx=0;
    do{
        __syncthreads();
      if(threadIdx.x == 0)
      {
        this->is_finshed[0] = true;

        for(int stride=this->stride[0]; stride <= this->stride[1]; stride++)
        {
          
          auto start_idx = stride;
          
          bool isDisable;
          if(iter_idx == 0)
          {
            isDisable = false;  
          }
          else
          {
           
            isDisable = disable[start_idx];  
          }
          
          
          auto score = scores[start_idx];
          
          if(score <= score_threshold || count == max_output_boxes)
          {
            break;
          }
          
          if(isDisable == false)
          {
            
            this->current_box[0] = boxes[start_idx];

            
            output_ids[count] = stride;
            count+=1;

            this->stride[0] = stride+1;
            if(this->stride[0] > this->stride[1])
            {
              this->is_finshed[0]=true;
            }
            else
            {
              this->is_finshed[0]=false;
            }
            break;
          }
        }
      }
      __threadfence();
      __syncthreads();
      
      if(this->is_finshed[0])
      {
          if(threadIdx.x==0)
          {
              
            this->count[0] = count;
            
          }
        return;
      }
      BOX current_box_selected(this->current_box[0]);

      for(int stride=this->stride[0]+threadIdx.x; stride <= this->stride[1]; stride+=blockDim.x)
      {
        
        auto start_idx = stride;

        
        BOX box(boxes[start_idx]);
        if(iter_idx == 0)
        {
          
          disable[start_idx] = false; 
        }
        
        bool is_disable = disable[start_idx];
       
        if(!is_disable && IOU_GE(current_box_selected,box,iou_threshold))
        {
          
          disable[start_idx] = true;
        }
        
      }
      iter_idx++;
      __threadfence();
      __syncthreads();
    

    }while(this->stride[0]<=this->stride[1]);


  }
private:
  OP0 boxes;//[sptial_dimension,4]
  OP1 scores;//[sptial_dimension]
  OP2 max_output_boxes_per_class;//[1]
  OP3 iou_threshold;//[1]
  OP4 score_threshold;//[1]
  OP5 sort_size;//[1]

  TensorBool disable;
  TensorIdx output_ids;
  TensorCount count;

  
  float4* current_box=nullptr;
  bool* is_finshed=nullptr;
  int* stride=nullptr;

  int* socres_dim_sizes;
};


template<typename OP0,
         typename OP1,
         typename OP2,
         typename OP3,
         typename OP4,
         typename OP5,
         typename TensorBool,
         typename TensorIdx,
         typename TensorCount,
         int center_point_box=true>
__device__ auto NonMaxSuppression(
                                  OP0 boxes, //tensor
                                  OP1 scores, //tensor
                                  OP2 max_output_boxes_per_class, //scalar
                                  OP3 iou_threshold, //scalar
                                  OP4 score_threshold,//scalar
                                  OP5 op5,//scalar
                                  TensorBool disable, //tensor
                                  TensorIdx output_ids,//tensor
                                  TensorCount count, //ptr one num 
                                  int* socres_dim_sizes
                                  )
{
  return NonMaxSuppressionOp<OP0,OP1,OP2,OP3,OP4,OP5,TensorBool,TensorIdx,TensorCount,center_point_box>
        (boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold,op5,disable,output_ids,count,socres_dim_sizes);
}

template<int center_point_box=0,int sorted=1>
__device__  void NonMaxSuppression_device_func(
                               float4* boxes_buffer,
                               float* scores_buffer,
                               int* max_output_boxes_per_class_buffer,
                               float* iou_threshold_buffer,
                               float* scores_threshold_buffer,
                               int box_class_num, //80
                               int B, //1
                               int C, //80
                               int box_s, 
                               int scores_s, 
                              bool* is_disable_buffer,
                              int* boxIds_buffer,
                              int* count_buffer,
                               int* sort_size_buffer=nullptr)
{
  int scores_dim_sizes[3]={B,C,scores_s}; 
  int box_dim_sizes[3]={B,box_class_num,box_s}; 

  
   float iou_threshold = iou_threshold_buffer[0];
   float scores_threshold = scores_threshold_buffer[0];

  
  
  int count_dim_sizes[2]={B,C};//(1,80)
  
  int batchIdx = blockIdx.x / C;
  
  int classIdx = blockIdx.x % C;

  int box_class_idx=(box_class_num==1?0:classIdx);

 
   float4& boxes_sub_buffer = boxes_buffer[(batchIdx*box_class_num*box_s + box_class_idx*box_s + 0)]; //{B,box_class_num,box_s}
  
   float& scores_sub_buffer = scores_buffer[(batchIdx*C*scores_s + classIdx*scores_s + 0)]; //{B,C,scores_s}
  
  
   int& max_output_boxes_per_class_sub_buffer = max_output_boxes_per_class_buffer[0]; 

  

 
  int& boxIds_sub_buffer = boxIds_buffer[(batchIdx*box_class_num*box_s + classIdx*box_s + 0)]; //{B,box_class_num,box_s}
  
  bool& is_disable_sub_buffer = is_disable_buffer[(batchIdx*box_class_num*box_s,classIdx*box_s + 0)]; //{B,box_class_num,box_s}
 
  int& count_sub_buffer=count_buffer[(batchIdx*C + classIdx)]; //{B,C}


  
  int boxes_dim_sizes[1] = {box_s}; //{B,box_class_num,box_s}
 
  int socres_dim_sizes[1] = {scores_s}; //{B,C,scores_s}


 
   float4* boxes_sub = &boxes_sub_buffer;
 
   float* scores_sub = &scores_sub_buffer;
  

  
   int max_output_boxes_per_class_sub = max_output_boxes_per_class_sub_buffer; 
  

  
  int* boxIds_sub = &boxIds_sub_buffer;
 
  bool* is_disable_sub = &is_disable_sub_buffer;
  
  int* count = &count_sub_buffer;
  int sort_size=-1;
  if(sort_size_buffer!=nullptr)
    sort_size = sort_size_buffer[batchIdx*C+classIdx];

  auto non = NonMaxSuppression<
                                decltype(boxes_sub),
                                decltype(scores_sub),
                                decltype(max_output_boxes_per_class_sub),
                                decltype(iou_threshold),
                                decltype(scores_threshold),
                                decltype(sort_size),
                                decltype(is_disable_sub),
                                decltype(boxIds_sub),
                                decltype(count),
                                center_point_box>(boxes_sub, //tensor
                                                  scores_sub, //tensor
                                                  max_output_boxes_per_class_sub, //scalar
                                                  iou_threshold, //scaler
                                                  scores_threshold, //scalar
                                                  sort_size, //scalar
                                                  is_disable_sub, //tensor
                                                  boxIds_sub, //tensor
                                                  count, //ptr one num
                                                  socres_dim_sizes);

  non.template AutoLoad(threadIdx.x);
}

extern "C" __global__  void NonMaxSuppression_device_func_global(
                               float4* boxes_buffer,
                               float* scores_buffer,
                               int* max_output_boxes_per_class_buffer,
                               float* iou_threshold_buffer,
                               float* scores_threshold_buffer,
                               int box_class_num, int B, int C,
                               int box_s,
                               int scores_s,
                              bool* is_disable_buffer,
                              int* boxIds_buffer,
                              int* count_buffer,
                               int* sort_size_buffer=nullptr)
{
  NonMaxSuppression_device_func<0,1>(boxes_buffer,
                               scores_buffer,
                                max_output_boxes_per_class_buffer,
                               iou_threshold_buffer,
                               scores_threshold_buffer,
                               box_class_num, B, C,
                               box_s,
                               scores_s,
                               is_disable_buffer,
                               boxIds_buffer,
                               count_buffer,
                               sort_size_buffer);
}
)code";

char yolov3_box_src_file[] = R"code(
    #include <cuda_runtime.h>

#define MIN(a,b) ((a)<(b) ? (a):(b))

template<typename T0>
__device__ auto sigmoid(T0 t0)
{
	return 1/(1 + expf(-t0));
}

template<typename T0>
__device__ auto rcp(T0 t0)
{
	return 1.0f/t0;
}

__device__ auto BoxesGetFeats(const float4* feats_buffer,
															const float* anchors_buffer,
															const int* input_shape_buffer,
															const int tidx_,
															const int b_idx, const int h_idx, const int w_idx,const int anchors_idx,
															const int H,const int W)
{
	//stage1: BoxesGetFeats
	float4 out_value;

	int grid_h = H;
	int grid_w = W;
	float4 value = feats_buffer[tidx_]; 

	out_value.x=(sigmoid(value.y)+h_idx) * rcp(grid_h*1.0f);
	out_value.y=(sigmoid(value.x)+w_idx) * rcp(grid_w*1.0f);

	out_value.z=expf(value.w); 
	out_value.z=out_value.z*anchors_buffer[anchors_idx*2 + 1]; //load anchors'h
	out_value.z=out_value.z/input_shape_buffer[1];//h,w so load 0

	
	out_value.w=expf(value.z); //thread2, load(x,y,w,h)-->h
	out_value.w=out_value.w*anchors_buffer[anchors_idx*2]; //load anchors'h
	out_value.w=out_value.w/input_shape_buffer[2];//h,w so load 0

	return out_value;
}

__device__ auto CorrectBoxes(float4 box_value,
														 const int* image_shape_buffer,
														 const int* input_shape_buffer,
														 const int b_idx)
{
	float4 out_value=box_value;

	auto image_shape_h=image_shape_buffer[b_idx*2];
	auto image_shape_w=image_shape_buffer[b_idx*2 + 1];

	auto input_h=input_shape_buffer[1];
	auto input_w=input_shape_buffer[2];

	auto scale=MIN(input_h*1.0f/image_shape_h,input_w*1.0f/image_shape_w);
	int new_shape_h=image_shape_h*scale+0.5f;
	int new_shape_w=image_shape_w*scale+0.5f;
	auto offset_h=(input_h-new_shape_h)/2.0f/input_h;
	auto offset_w=(input_w-new_shape_w)/2.0f/input_w;
	
	auto scale_h=input_h*1.0f/new_shape_h;
	auto scale_w=input_w*1.0f/new_shape_w;



	box_value.x = (box_value.x - offset_h) * scale_h;
				
	auto box_value_h = box_value.z*scale_h; 
	
	out_value.x = (box_value.x - box_value_h/2)*image_shape_h;
	out_value.z = (box_value.x + box_value_h/2)*image_shape_h;

	//thread1 compute x1
	box_value.y = (box_value.y - offset_w) * scale_w;
	
	auto box_value_w = box_value.w*scale_w; 
	
	out_value.y = (box_value.y - box_value_w/2)*image_shape_w;
	out_value.w = (box_value.y + box_value_w/2)*image_shape_w;

	return out_value;
}

__device__ void dl_boxes_float4_device_func(
		const float4* feats_buffer, 
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	for(int tidx_ = threadIdx.x + blockDim.x * blockIdx.x; tidx_ < B*H*W*anchor_num; tidx_ += blockDim.x*gridDim.x)
	{
		int tidx = tidx_*4;
		auto box_class_idx= 0; //tidx % 4; // 1,2 is x,y  3,4 is w,h
		int prev_idx=tidx / 4; //which (x,y,w,h), total is H*W*anchor_num every batch
		auto out_num_idx=prev_idx%(anchor_num*H*W);//this batch's which (x,y,w,h)
		auto b_idx=prev_idx/(anchor_num*H*W);//which batch

		auto anchor_idx=out_num_idx%anchor_num; //total(H*W*anchor_num)'s which anchor
		auto w_idx=(out_num_idx/anchor_num)%W; //total(H*W)'s which W
		auto h_idx=(out_num_idx/anchor_num)/W;

		float4 out_value = BoxesGetFeats(feats_buffer,
																		 anchors_buffer,
																		 input_shape_buffer,
																		 tidx_,
																		 b_idx,h_idx,w_idx,anchor_idx,
																		 H,W);
		out_value = CorrectBoxes(out_value,
														 image_shape_buffer,
														 input_shape_buffer,
														 b_idx);

		boxes_buffer[tidx_] = out_value;

	}
}

extern "C" __global__  void dl_boxes_float4_device_func_global(
		const float4* feats_buffer, 
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func(
		feats_buffer, 
		anchors_buffer,//[3,2]
		input_shape_buffer, //shape == (4,) nhwc || nchw
		image_shape_buffer, //shape == (none,2) hw
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}

__device__  void dl_boxes_float4_device_func_kernel(
		const float4* feats_buffer, 
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func(
		feats_buffer, 
		anchors_buffer,//[3,2]
		input_shape_buffer, //shape == (4,) nhwc || nchw
		image_shape_buffer, //shape == (none,2) hw
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}









__device__ auto BoxesGetFeatsAligned(const float4* feats_buffer,
															const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
															const float* anchors_buffer,
															const int* input_shape_buffer,
															const int tidx_,
															const int b_idx, const int h_idx, const int w_idx,const int anchors_idx,
															const int H,const int W)
{
	//stage1: BoxesGetFeats
	int f4_feats_s3 = feats_s3;
	int f4_feats_s2 = feats_s2/4;
	int f4_feats_s1 = feats_s1/4;
	int f4_feats_s0 = feats_s0/4;

	float4 out_value;

	int grid_h = H;
	int grid_w = W;
	// float4 value = feats_buffer[tidx_]; //(n,h,w,12) --> (n,h,w,3,4) --> (float4)(n,h,w,3) -->(float4 aligned)(n,h,w,4)
	float4 value = feats_buffer[b_idx*H*W*4 + h_idx*W*4 + w_idx*4 + anchors_idx];
	// float4 value = feats_buffer[b_idx*f4_feats_s0 + h_idx*f4_feats_s1 + w_idx*f4_feats_s2 + anchors_idx*f4_feats_s3];

	out_value.x=(sigmoid(value.y)+h_idx) * rcp(grid_h*1.0f);
	out_value.y=(sigmoid(value.x)+w_idx) * rcp(grid_w*1.0f);

	out_value.z=expf(value.w); //thread2, load(x,y,w,h)-->h
	out_value.z=out_value.z*anchors_buffer[anchors_idx*2 + 1]; //load anchors'h
	out_value.z=out_value.z/input_shape_buffer[1];//h,w so load 0

	
	out_value.w=expf(value.z); //thread2, load(x,y,w,h)-->h
	out_value.w=out_value.w*anchors_buffer[anchors_idx*2]; //load anchors'h
	out_value.w=out_value.w/input_shape_buffer[2];//h,w so load 0

	return out_value;
}

__device__ void dl_boxes_float4_device_func_aligned(
		const float4* feats_buffer, //typename Tensor0::DATA_TYPE* feats_buffer,//[None,h,w,12]
		const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	for(int tidx_ = threadIdx.x + blockDim.x * blockIdx.x; tidx_ < B*H*W*anchor_num; tidx_ += blockDim.x*gridDim.x)
	{
		int tidx = tidx_*4;
		auto box_class_idx= 0; //tidx % 4; // 1,2 is x,y  3,4 is w,h
		int prev_idx=tidx / 4; //which (x,y,w,h), total is H*W*anchor_num every batch
		auto out_num_idx=prev_idx%(anchor_num*H*W);//this batch's which (x,y,w,h)
		auto b_idx=prev_idx/(anchor_num*H*W);//which batch

		auto anchor_idx=out_num_idx%anchor_num; //total(H*W*anchor_num)'s which anchor
		auto w_idx=(out_num_idx/anchor_num)%W; //total(H*W)'s which W
		auto h_idx=(out_num_idx/anchor_num)/W;

		float4 out_value = BoxesGetFeatsAligned(feats_buffer,
																		 feats_s0,feats_s1,feats_s2,feats_s3,
																		 anchors_buffer,
																		 input_shape_buffer,
																		 tidx_,
																		 b_idx,h_idx,w_idx,anchor_idx,
																		 H,W);
		out_value = CorrectBoxes(out_value,
														 image_shape_buffer,
														 input_shape_buffer,
														 b_idx);

		boxes_buffer[tidx_] = out_value;

	}
}

extern "C" __global__  void dl_boxes_float4_device_func_global_aligned(
		const float4* feats_buffer, //typename Tensor0::DATA_TYPE* feats_buffer,//[None,h,w,12]
		const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func_aligned(
		feats_buffer, //typename Tensor0::DATA_TYPE* feats_buffer,//[None,h,w,12]
		feats_s0,feats_s1,feats_s2,feats_s3,
		anchors_buffer,//[3,2]
		input_shape_buffer, //shape == (4,) nhwc || nchw
		image_shape_buffer, //shape == (none,2) hw
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}

__device__  void dl_boxes_float4_device_func_kernel_aligned(
		const float4* feats_buffer, 
		const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
		const float* anchors_buffer,
		const int* input_shape_buffer, 
		const int* image_shape_buffer,
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func_aligned(
		feats_buffer, 
		feats_s0,feats_s1,feats_s2,feats_s3,
		anchors_buffer,
		input_shape_buffer, 
		image_shape_buffer, 
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}

)code";
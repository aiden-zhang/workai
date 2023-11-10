#ifndef TOPK_H_
#define TOPK_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

#include <stdio.h>
#include <algorithm>
#include <vector>

#define MAX(a,b)((a)>(b)?(a):(b))

#define MAX_4(a,b,c,d) MAX(MAX(a,b),MAX(c,d))

#define CheckError(err)                   \
	if (err != cudaSuccess){              \
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));\
        exit(1);                          \
    };

typedef half float16;
typedef half2 float16x2;
typedef float2 float16x4;

struct float32x16
{
	float4 x;
	float4 y;
	float4 z;
	float4 w;
};
struct float16x16
{
	float16x4 x;
	float16x4 y;
	float16x4 z;
	float16x4 w;
};

namespace{

template<typename>
struct VectorType
{
};
template<>
struct VectorType<float>
{
	typedef float4 Type4D;
	typedef float2 Type2D;
};

template<>
struct VectorType<float16>
{
	typedef float2 Type4D;
	typedef float16x2 Type2D;
};

template<>
struct VectorType<int>
{
	typedef int4 Type4D;
	typedef int2 Type2D;
};


template<int value>
__device__ constexpr int Log2(){
	for(int i=31;i>=0;i--){
	 	if (1==(value>>i)){
	 		return i;
	 	}
	}
	return -1;
}
template<typename DataType,typename Idx,bool IsAscending=true>
__device__ void sort(DataType& a,Idx& a_idx,DataType& b,Idx& b_idx){
	if((a>b && IsAscending) || (a<b && (IsAscending==false)) ){
		
		DataType tmp=a;
		Idx tmp_idx=a_idx;
		a=b;
		a_idx=b_idx;
		b=tmp;
		b_idx=tmp_idx;
	}
}

template<typename DataType,typename Idx,bool IsAscending=true,int BLOCK_SIZE=32,int tile_num=1,int K=128>
__device__ void  BitonicSortSharedPart(DataType* eles,Idx* idxs){//ele is shared_memory size== BLOCK_SIZE*2

	// #pragma unroll
	for(int log2_stride=0;log2_stride< Log2<(K)>();log2_stride++){

		int log2_rev_size_local=log2_stride+1;
		// #pragma unroll
		for (int log2_stride_ = log2_stride; log2_stride_>=0; log2_stride_--){
		    int stride_=1<<log2_stride_;

		    __syncthreads();
		  	
		  	// #pragma unroll
		    for(int tid=0;tid<tile_num;tid++){
		    	 if(tid*BLOCK_SIZE*2>=K){
		    	 	break;
		    	 }
		    	 int tid_x=tid*BLOCK_SIZE+threadIdx.x;
				 int bid=tid_x>>log2_stride_;

	      		 int tidx=(tid_x&((1<<log2_stride_)-1))+bid*stride_*2;

	      		 DataType ele0=eles[tidx];
	      		 Idx idx0=idxs[tidx];
		      	 DataType ele1=eles[tidx+stride_];
		      	 Idx idx1=idxs[tidx+stride_];
		      		 
		      	 if((tidx>>log2_rev_size_local) &1){
		      	 	sort<DataType,Idx,(!IsAscending)>(ele0,idx0,ele1,idx1);
		      	 }else{
		      	 	sort<DataType,Idx,IsAscending>(ele0,idx0,ele1,idx1);
		      	 }
		      	 eles[tidx]=ele0;
		      	 idxs[tidx]=idx0;
		      	 eles[tidx+stride_]=ele1;
		      	 idxs[tidx+stride_]=idx1;

	      	}
		}
	}
	//__syncthreads();
	
}



template<bool flag>
struct Top_K
{
template<typename DataType,typename Idx,bool IsAscending=true,int BLOCK_SIZE=32,int tile_num=1,
		int K=128,bool is_sorted=false,int valid_num=-1>
__device__ inline static void  BitonicSortSharedTopK(DataType* eles,Idx* idxs){//ele is shared_memory size== BLOCK_SIZE*2

	constexpr int mask_private_sort=K*2-1;
	// #pragma unroll
	for(int log2_stride=0;log2_stride< Log2<(K)>()+1;log2_stride++){

		int log2_rev_size_local=log2_stride+1;
		// #pragma unroll
		for (int log2_stride_ = log2_stride; log2_stride_>=0; log2_stride_--){
		    int stride_=1<<log2_stride_;

		    __syncthreads();
		    // #pragma unroll
		    for(int tid=0;tid<tile_num;tid++){
		    	 if(tid*BLOCK_SIZE*2>=valid_num){
		    	 	break;
		    	 }
		    	 int tid_x=tid*BLOCK_SIZE+threadIdx.x;
				 int bid=tid_x>>log2_stride_;
	      		 int tidx=(tid_x&((1<<log2_stride_)-1))+bid*stride_*2;

	      		 DataType ele0=eles[tidx];
	      		 Idx idx0=idxs[tidx];
		      	 DataType ele1=eles[tidx+stride_];
		      	 Idx idx1=idxs[tidx+stride_];      	

		      	 if((((tidx&mask_private_sort)>>log2_rev_size_local) &1)){
		      	 	sort<DataType,Idx,(!IsAscending)>(ele0,idx0,ele1,idx1);
		      	 }else{
		      	 	sort<DataType,Idx,IsAscending>(ele0,idx0,ele1,idx1);
		      	 }
		      	
		      	 if(log2_stride==Log2<(K)>() && log2_stride_==log2_stride){
		      	 	 int current_tidx=tidx/(K*2)*K+tidx%K;
		      	 	 __syncthreads();
		      	 	 if(((tidx&mask_private_sort)) <K){
					   eles[current_tidx]=ele0;
			      	   idxs[current_tidx]=idx0;
					 }
					 __syncthreads();
				 }else{
				 	 eles[tidx]=ele0;
			      	 idxs[tidx]=idx0;
			      	 eles[tidx+stride_]=ele1;
			      	 idxs[tidx+stride_]=idx1;
				 }
	      	}
	      	if(log2_stride==Log2<(K)>()){
				break;
			} 	
		}
	}

	constexpr int TILE_NUM=tile_num>=2? tile_num/2:1;
	
	constexpr int next_valid_num=valid_num>>1;
    __syncthreads();

	Top_K<(next_valid_num>K)>::template BitonicSortSharedTopK<DataType,Idx,IsAscending,
			BLOCK_SIZE,TILE_NUM,K,is_sorted,next_valid_num>(eles,idxs);
}

};

template<>
struct Top_K<false>
{
	template<typename DataType,typename Idx,bool IsAscending=true,int BLOCK_SIZE=32,int tile_num=1,
			int K=128,bool is_sorted=false,int valid_num=-1>
	__device__ inline static void  BitonicSortSharedTopK(DataType* eles,Idx* idxs){//ele is shared_memory size== BLOCK_SIZE*2	
		if(is_sorted)
			BitonicSortSharedPart<DataType,Idx,IsAscending,BLOCK_SIZE,tile_num,K>(eles,idxs);
	}
};


/*
This TopkOperation is for run
when run we need a valid_num to reduce thread number
*/
template<bool IsSorted,typename DataType,typename DataIdxType>
struct TopkOperation
{
using Type4D=typename VectorType<DataType>::Type4D ;	
using Index4D=typename VectorType<DataIdxType>::Type4D;

template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16>
__device__ static void Run(DataType* in,DataIdxType* in_idx,DataType* out,DataIdxType* out_idx,int valid_num=99999999){
	__shared__ DataType sm_in[BLOCK_SIZE*TILE_NUM];
	__shared__ DataIdxType sm_in_idx[BLOCK_SIZE*TILE_NUM];

	// #pragma unroll
	for(int tile_idx=0;tile_idx<TILE_NUM;tile_idx++){
		int in_index=blockIdx.x*BLOCK_SIZE*TILE_NUM+threadIdx.x+BLOCK_SIZE*tile_idx;

		sm_in[threadIdx.x+BLOCK_SIZE*tile_idx]=
			in_index<valid_num ? in[in_index] :-99999999999999.0f;

		sm_in_idx[threadIdx.x+BLOCK_SIZE*tile_idx]=
			in_idx[blockIdx.x*BLOCK_SIZE*TILE_NUM+threadIdx.x+BLOCK_SIZE*tile_idx];
	}


	constexpr bool flag=BLOCK_SIZE*TILE_NUM>K;

	Top_K<flag>::template BitonicSortSharedTopK<DataType,DataIdxType,
					false,BLOCK_SIZE,TILE_NUM/2,K,IsSorted,BLOCK_SIZE*TILE_NUM>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=BLOCK_SIZE){
		out[blockIdx.x*K+tidx]=sm_in[tidx];
		out_idx[blockIdx.x*K+tidx]=sm_in_idx[tidx];

		// printf("heelo%d %f %d\n",tidx,sm_in[tidx],sm_in_idx[tidx]);
	}
}


template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16>
__device__ static void Run(DataType* in,DataType* out,DataIdxType* out_idx,int valid_num=99999999){
	__shared__ DataType sm_in[BLOCK_SIZE*TILE_NUM];
	__shared__ DataIdxType sm_in_idx[BLOCK_SIZE*TILE_NUM];

	// #pragma unroll
	for(int tile_idx=0;tile_idx<TILE_NUM;tile_idx++){

		int in_index=blockIdx.x*BLOCK_SIZE*TILE_NUM+threadIdx.x+BLOCK_SIZE*tile_idx;

		sm_in[threadIdx.x+BLOCK_SIZE*tile_idx]=in_index<valid_num ? in[in_index] :DataType(-99999999999999.0f);

		sm_in_idx[threadIdx.x+BLOCK_SIZE*tile_idx]=
			blockIdx.x*BLOCK_SIZE*TILE_NUM+threadIdx.x+BLOCK_SIZE*tile_idx;
	}


	constexpr bool flag=BLOCK_SIZE*TILE_NUM>K;

	Top_K<flag>::template BitonicSortSharedTopK<DataType,DataIdxType,
					false,BLOCK_SIZE,TILE_NUM/2,K,IsSorted,BLOCK_SIZE*TILE_NUM>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=BLOCK_SIZE){
		out[blockIdx.x*K+tidx]=sm_in[tidx];
		out_idx[blockIdx.x*K+tidx]=sm_in_idx[tidx];

		// printf("heelo%d %f %d\n",tidx,sm_in[tidx],sm_in_idx[tidx]);
	}
}

};

template<int idx>
struct GetArgsHelper
{	
	template<typename T,typename...Ts>
	__device__ static auto Get(T arg0,Ts... args){

		return GetArgsHelper<idx-1>::template Get<Ts...>(args...);
	}
};
template<>
struct GetArgsHelper<0>
{
	template<typename T,typename...Ts>
	__device__ static auto Get(T arg0,Ts... args){
		return arg0;
	}
};
template<int ArgNum>
struct GetArgsImpl
{
	
};
template<>
struct GetArgsImpl<1>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	
	return GetArgsHelper<0>::template Get<Ts...>(args...);
}
};
template<>
struct GetArgsImpl<2>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		default:return GetArgsHelper<1>::template Get<Ts...>(args...);
	}

}
};
template<>
struct GetArgsImpl<3>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		default:return GetArgsHelper<2>::template Get<Ts...>(args...);
	}
}
};
template<>
struct GetArgsImpl<4>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		case 2:return GetArgsHelper<2>::template Get<Ts...>(args...);
		default:return GetArgsHelper<3>::template Get<Ts...>(args...);
	}
}
};

template<>
struct GetArgsImpl<5>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		case 2:return GetArgsHelper<2>::template Get<Ts...>(args...);
		case 3:return GetArgsHelper<3>::template Get<Ts...>(args...);
		default:return GetArgsHelper<4>::template Get<Ts...>(args...);
	}
}
};
template<>
struct GetArgsImpl<6>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		case 2:return GetArgsHelper<2>::template Get<Ts...>(args...);
		case 3:return GetArgsHelper<3>::template Get<Ts...>(args...);
		case 4:return GetArgsHelper<4>::template Get<Ts...>(args...);
		default:return GetArgsHelper<5>::template Get<Ts...>(args...);
	}
}
};
template<>
struct GetArgsImpl<7>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		case 2:return GetArgsHelper<2>::template Get<Ts...>(args...);
		case 3:return GetArgsHelper<3>::template Get<Ts...>(args...);
		case 4:return GetArgsHelper<4>::template Get<Ts...>(args...);
		case 5:return GetArgsHelper<5>::template Get<Ts...>(args...);
		default:return GetArgsHelper<6>::template Get<Ts...>(args...);
	}
}
};

template<>
struct GetArgsImpl<8>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		case 2:return GetArgsHelper<2>::template Get<Ts...>(args...);
		case 3:return GetArgsHelper<3>::template Get<Ts...>(args...);
		case 4:return GetArgsHelper<4>::template Get<Ts...>(args...);
		case 5:return GetArgsHelper<5>::template Get<Ts...>(args...);
		case 6:return GetArgsHelper<6>::template Get<Ts...>(args...);
		default:return GetArgsHelper<7>::template Get<Ts...>(args...);
	}
}
};


template<>
struct GetArgsImpl<9>
{
template<typename... Ts>
__device__ static auto GetArgs(int idx,Ts... args){
	switch(idx){
		case 0:return GetArgsHelper<0>::template Get<Ts...>(args...);
		case 1:return GetArgsHelper<1>::template Get<Ts...>(args...);
		case 2:return GetArgsHelper<2>::template Get<Ts...>(args...);
		case 3:return GetArgsHelper<3>::template Get<Ts...>(args...);
		case 4:return GetArgsHelper<4>::template Get<Ts...>(args...);
		case 5:return GetArgsHelper<5>::template Get<Ts...>(args...);
		case 6:return GetArgsHelper<6>::template Get<Ts...>(args...);
		case 7:return GetArgsHelper<7>::template Get<Ts...>(args...);
		default:return GetArgsHelper<8>::template Get<Ts...>(args...);
	}
}
};



template<typename... Ts>
__device__ auto get_args(int idx,Ts... args){
	return GetArgsImpl<sizeof...(args)>::GetArgs(idx,args...);
}

template<typename TV,typename T>
struct GetVectorCond
{
	static __device__ auto get(TV* buffer,int address,int length){
		TV value;
		if((address+1)*4>length){ //only one ddr
			T* tmp_value=(T*)&buffer[address];
			if(address*4+3>length){
				value.w=T(-99999999.0f);
			}else{
				value.w=tmp_value[3];
			}
			if(address*4+2>length){
				value.z=T(-99999999.0f);
			}else{
				value.z=tmp_value[2];
			}
			if(address*4+1>length){
				value.y=T(-99999999.0f);
			}else{
				value.y=tmp_value[1];
			}
			if(address*4+0>length){
				value.x=T(-99999999.0f);
			}else{
				value.x=tmp_value[0];
			}

		}else{
			value=buffer[address];
		}

		return value;
	}
};

template<>
struct GetVectorCond<float16x4,float16>
{
	static __device__ auto get(float16x4* buffer,int address,int length){
		float16x4 value;

		float16* tmp_buffer=(float16*)buffer;
		half2 pre_value;
		half2 now_value;
		if((address+1)*4>length){
			if(address*4>=length){
				pre_value.x=float16(-99999.0f);
			}else{
				pre_value.x=tmp_buffer[address*4];
			}
			if(address*4+1>=length){
				pre_value.y=float16(-99999.0f);
			}else{
				pre_value.y=tmp_buffer[address*4+1];
			}

			if(address*4+2>=length){
				now_value.x=float16(-99999.0f);
			}else{
				now_value.x=tmp_buffer[address*4+2];
			}
			if(address*4+3>=length){
				now_value.y=float16(-99999.0f);
			}else{
				now_value.y=tmp_buffer[address*4+3];
			}
			value.x=*((float*)&(pre_value));
			value.y=*((float*)&(now_value));
		}else{
			value=buffer[address];
		}
		return value;
	}
};
/*
This TopkOperation is for online compile cu to fatbin file
So don't need valid_num
*/
template<bool IsSorted>
struct TopkOperation<IsSorted,float4,int>
{

template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE, float* out,int* out_idx,int* select_in_idx,DataTypeIns... ins){

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*4];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*4];

	int4* sm_in_idx_4=(int4*)sm_in_idx;
	
	constexpr int NEW_TILE_NUM = BLOCK_SIZE==1024? TILE_NUM*2: TILE_NUM;
	constexpr int NEW_BLOCK_SIZE = BLOCK_SIZE==1024? 512: BLOCK_SIZE;

	#pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[threadIdx.x+tile_idx*NEW_BLOCK_SIZE];
		
		float4* value_block=((float4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		
		//warning one use one ddr
		float4 value=GetVectorCond<float4,float>::get(value_block,address,TILE_BLOCK_SIZE);
		
		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=value;

		int4 address_4;
		address_4.x=address*4;
		address_4.y=address*4+1;
		address_4.z=address*4+2;
		address_4.w=address*4+3;

	
		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=NEW_BLOCK_SIZE*NEW_TILE_NUM*4>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,
		NEW_BLOCK_SIZE*NEW_TILE_NUM*4>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[blockIdx.x*K+tidx]=sm_in[tidx];
		out_idx[blockIdx.x*K+tidx]=sm_in_idx[tidx];
	}

}


template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE,float* out,int* out_idx,int* select_in_idx,int* input_index,DataTypeIns... ins){

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*4];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*4];

	int4* sm_in_idx_4=(int4*)sm_in_idx;
	
	constexpr int NEW_TILE_NUM = BLOCK_SIZE==1024? TILE_NUM*2: TILE_NUM;
	constexpr int NEW_BLOCK_SIZE = BLOCK_SIZE==1024? 512: BLOCK_SIZE;

	#pragma unroll
	for(int tile_idx=0;tile_idx<TILE_NUM;tile_idx++){

		int address=select_in_idx[blockIdx.x*NEW_BLOCK_SIZE*NEW_TILE_NUM
				+threadIdx.x+tile_idx*NEW_BLOCK_SIZE];
		
		float4* value_block=((float4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		//warning one use one ddr
		float4 value=GetVectorCond<float4,float>::get(value_block,address,TILE_BLOCK_SIZE);
		// printf("%f %f %f %f \n", value.x,value.y,value.z,value.w);
		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=value;

		int4 address_4=((int4*)input_index)[address];

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=NEW_TILE_NUM*NEW_TILE_NUM*4>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,
			IsSorted,NEW_BLOCK_SIZE*NEW_TILE_NUM*4>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[blockIdx.x*K+tidx]=sm_in[tidx];
		out_idx[blockIdx.x*K+tidx]=sm_in_idx[tidx];
	}

}
};



template<bool IsSorted>
struct TopkOperation<IsSorted,float32x16,int>
{

template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE, float* out,int* out_idx,int* select_in_idx,
	DataTypeIns... ins){

	// printf("%d %f\n",threadIdx.x,BLOCK_SIZE*TILE_NUM*16);

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*16];
	float4* sm_in_4=(float4*)sm_in;

    __shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*16+1024*4];

	int4* sm_in_idx_4=(int4*)sm_in_idx;

	//TODO: mybe more readable
	constexpr int NEW_BLOCK_SIZE = K*4==1024 ? 512 : BLOCK_SIZE * 4; //threads
	constexpr int NEW_TILE_NUM = K*4==1024 ? TILE_NUM * 2 : TILE_NUM;

	// #pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[(threadIdx.x+tile_idx*NEW_BLOCK_SIZE)>>2]*4+(threadIdx.x&3);


		
		float4* value_block=((float4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		//warning one use one ddr
		float4 value=GetVectorCond<float4,float>::get(value_block,address,TILE_BLOCK_SIZE);

		

		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=value;

		auto tmp_idx=threadIdx.x+NEW_BLOCK_SIZE*tile_idx;



		int4 address_4;
		address_4.x=address*4;
		address_4.y=address*4+1;
		address_4.z=address*4+2;
		address_4.w=address*4+3;

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=BLOCK_SIZE*TILE_NUM*16>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,BLOCK_SIZE*TILE_NUM*16>(sm_in,sm_in_idx);
    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[tidx]=sm_in[tidx];
		out_idx[tidx]=sm_in_idx[tidx];
	}

}


template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE,float* out,int* out_idx,
		int* select_in_idx,int* input_index,DataTypeIns... ins){

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*16];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*16];

	int4* sm_in_idx_4=(int4*)sm_in_idx;

	//TODO: mybe more readable
	constexpr int NEW_BLOCK_SIZE = K*4==1024 ? 512 : BLOCK_SIZE * 4; //threads
	constexpr int NEW_TILE_NUM = K*4==1024 ? TILE_NUM * 2 : TILE_NUM;

	// #pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[(threadIdx.x+tile_idx*NEW_BLOCK_SIZE)>>2]*4+(threadIdx.x&3);

		
		float4* value_block=((float4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		
		//warning one use one ddr
		float4 value=GetVectorCond<float4,float>::get(value_block,address,TILE_BLOCK_SIZE);

		// printf("%f %f %f %f \n", value.x,value.y,value.z,value.w);
		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=value;

		int4 address_4=((int4*)input_index)[address];

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=BLOCK_SIZE*TILE_NUM*16>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,BLOCK_SIZE*TILE_NUM*16>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[tidx]=sm_in[tidx];
		out_idx[tidx]=sm_in_idx[tidx];
	}

}
};


template<bool IsSorted>
struct TopkOperation<IsSorted,float16x4,int>
{

template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE, float16* out,int* out_idx,int* select_in_idx,DataTypeIns... ins){

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*4];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*4];

	int4* sm_in_idx_4=(int4*)sm_in_idx;


	constexpr int NEW_TILE_NUM = BLOCK_SIZE==1024? TILE_NUM*2: TILE_NUM;
	constexpr int NEW_BLOCK_SIZE = BLOCK_SIZE==1024? 512: BLOCK_SIZE;

	#pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[threadIdx.x+tile_idx*NEW_BLOCK_SIZE];

		

		float16x4* value_block=((float16x4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		
		//warning only one ddr
		float16x4 value=GetVectorCond<float16x4,float16>::get(value_block,address,TILE_BLOCK_SIZE);

		float value_x=value.x;

		float16x2* value_front=(float16x2*)&(value_x);

		float4 c;
		c.x=float(value_front->x);
		c.y=float(value_front->y);

		value_x=value.y;
		value_front=(float16x2*)&(value_x);
		c.z=float(value_front->x);
		c.w=float(value_front->y);

		// printf("%d %f %f %f %f\n",threadIdx.x,c.x,c.y,c.z,c.w);

		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=c;

		int4 address_4;
		address_4.x=address*4;
		address_4.y=address*4+1;
		address_4.z=address*4+2;
		address_4.w=address*4+3;

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=BLOCK_SIZE*TILE_NUM*4>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,BLOCK_SIZE*TILE_NUM*4>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[tidx]=float16(sm_in[tidx]);
		out_idx[tidx]=sm_in_idx[tidx];
	}

}


template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE,float* out,int* out_idx,int* select_in_idx,int* input_index,DataTypeIns... ins){

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*4];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*4];

	int4* sm_in_idx_4=(int4*)sm_in_idx;

	constexpr int NEW_TILE_NUM = BLOCK_SIZE==1024? TILE_NUM*2: TILE_NUM;
	constexpr int NEW_BLOCK_SIZE = BLOCK_SIZE==1024? 512: BLOCK_SIZE;

	#pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[threadIdx.x+tile_idx*NEW_BLOCK_SIZE];

		
		float16x4* value_block=((float16x4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		//warning only one ddr
		float16x4 value=GetVectorCond<float16x4,float16>::get(value_block,address,TILE_BLOCK_SIZE);


		float4 c;
		float tmp=value.x;

		float16x2* c_pre_half=(float16x2*)(&tmp);
		tmp=value.y;

		float16x2* c_back_half=(float16x2*)(&tmp);

		c.x=c_pre_half->x;
		c.y=c_pre_half->y;
		c.z=c_back_half->x;
		c.w=c_back_half->y;

		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=c;

		int4 address_4=((int4*)input_index)[address];;

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=BLOCK_SIZE*TILE_NUM*4>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,BLOCK_SIZE*TILE_NUM*4>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[tidx]=sm_in[tidx];
		out_idx[tidx]=sm_in_idx[tidx];
	}

}
};



template<bool IsSorted>
struct TopkOperation<IsSorted,float16x16,int>
{

template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>
__device__ static void Run(int TILE_BLOCK_SIZE, float16* out,int* out_idx,int* select_in_idx,DataTypeIns... ins){

	// printf("%d %f\n",threadIdx.x,BLOCK_SIZE*TILE_NUM*16);

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*16];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*16];

	int4* sm_in_idx_4=(int4*)sm_in_idx;

	constexpr int NEW_BLOCK_SIZE = K*4==1024 ? 512 : BLOCK_SIZE * 4; //threads
	constexpr int NEW_TILE_NUM = K*4==1024 ? TILE_NUM * 2 : TILE_NUM;

	#pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[(threadIdx.x+tile_idx*NEW_BLOCK_SIZE)>>2]*4+(threadIdx.x&3);

		
		float2* value_block=((float2*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		//warning one use one ddr
		float16x4 value=GetVectorCond<float16x4,float16>::get(value_block,address,TILE_BLOCK_SIZE);

		float value_x=value.x;

		float16x2* value_front=(float16x2*)&(value_x);

		float4 c;
		c.x=float(value_front->x);
		c.y=float(value_front->y);

		value_x=value.y;
		value_front=(float16x2*)&(value_x);
		c.z=float(value_front->x);
		c.w=float(value_front->y);

		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=c;

		int4 address_4;
		address_4.x=address*4;
		address_4.y=address*4+1;
		address_4.z=address*4+2;
		address_4.w=address*4+3;

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=BLOCK_SIZE*TILE_NUM*16>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,BLOCK_SIZE*TILE_NUM*16>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[tidx]=sm_in[tidx];
		out_idx[tidx]=sm_in_idx[tidx];
	}


}


template<int BLOCK_SIZE=128,int TILE_NUM=2,int K=16,typename... DataTypeIns>

__device__ static void Run(int TILE_BLOCK_SIZE,float* out,int* out_idx,
		int* select_in_idx,int* input_index,DataTypeIns... ins){

	__shared__ float sm_in[BLOCK_SIZE*TILE_NUM*16];
	float4* sm_in_4=(float4*)sm_in;

	__shared__ int sm_in_idx[BLOCK_SIZE*TILE_NUM*16];

	int4* sm_in_idx_4=(int4*)sm_in_idx;
	constexpr int NEW_BLOCK_SIZE = K*4==1024 ? 512 : BLOCK_SIZE * 4; //threads
	constexpr int NEW_TILE_NUM = K*4==1024 ? TILE_NUM * 2 : TILE_NUM;

	#pragma unroll
	for(int tile_idx=0;tile_idx<NEW_TILE_NUM;tile_idx++){

		int address=select_in_idx[(threadIdx.x+tile_idx*NEW_BLOCK_SIZE)>>2]*4+(threadIdx.x&3);


		
		float16x4* value_block=((float16x4*)get_args<DataTypeIns...>(address/((TILE_BLOCK_SIZE+3)/4),ins...));
		
		//warning one use one ddr
		float16x4 value=GetVectorCond<float16x4,float16>::get(value_block,address,TILE_BLOCK_SIZE);

		float value_x=value.x;

		float16x2* value_front=(float16x2*)&(value_x);

		float4 c;
		c.x=float(value_front->x);
		c.y=float(value_front->y);

		value_x=value.y;
		value_front=(float16x2*)&(value_x);
		c.z=float(value_front->x);
		c.w=float(value_front->y);

		sm_in_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=c;

		int4 address_4=((int4*)input_index)[address];

		sm_in_idx_4[threadIdx.x+NEW_BLOCK_SIZE*tile_idx]=address_4;
	}

	constexpr bool flag=BLOCK_SIZE*TILE_NUM*16>K;

	Top_K<flag>::template BitonicSortSharedTopK<float,int,
					false,NEW_BLOCK_SIZE,NEW_TILE_NUM*4/2,K,IsSorted,BLOCK_SIZE*TILE_NUM*16>(sm_in,sm_in_idx);

    __syncthreads();
	for(int tidx=threadIdx.x;tidx<K;tidx+=NEW_BLOCK_SIZE){
		out[tidx]=sm_in[tidx];
		out_idx[tidx]=sm_in_idx[tidx];
	}

}
};


template<typename T,int Reduce>
struct VectorTypeMap
{
	// typedef float4 type;
};
template<>
struct VectorTypeMap<float,4>
{
	typedef float4 type;
};
template<>
struct VectorTypeMap<float,2>
{
	typedef float2 type;
};
template<>
struct VectorTypeMap<int,2>
{
	typedef int2 type;

	__device__ static type make(int v){
		type c;
		c.x=v;
		c.y=v+1;
		return c;
	} 
};
template<>
struct VectorTypeMap<int,4>
{
	typedef int4 type;

	__device__ static type make(int v){
		type c;
		c.x=v;
		c.y=v+1;
		c.z=v+2;
		c.w=v+3;
		return c;
	} 
};

template<int K,typename DataType,int... ReduceNums>
struct TopkSelect
{
	__device__ TopkSelect(DataType* reduce_data_buffer){
		
	}
	__device__ __inline__ void Run(DataType* sm_value,int* sm_value_idx,int*  row_idx){


	}
};

template<typename T>
struct make_vector
{
	static auto make(float value){

		return 0.0f;
	}
};

template<>
struct make_vector<float4>
{
	static __device__ auto make(float value){
		float4 a;
		a.x=value;
		a.y=value;
		a.z=value;
		a.w=value;
		return a;
	}
};


template<>
struct make_vector<float2>
{
	static __device__  auto make(float value){
		float2 a;
		a.x=value;
		a.y=value;
		return a;
	}
};


template<int K,typename DataType,int Reduce,int OFFSET,int... ReduceNums>
struct TopkSelect<K,DataType,Reduce,OFFSET,ReduceNums...>
{	
	using DataTypeVect=typename VectorTypeMap<DataType,Reduce>::type;
	using DataIndexTypeVect=typename VectorTypeMap<int,Reduce>::type;
	constexpr static int DataNums=K*Reduce;

	__device__ TopkSelect(DataType* reduce_data_buffer){
		this->reduce_data_buffer=reduce_data_buffer;
	}

	__device__ __inline__ void Run(DataType* sm_value,int* sm_value_idx,int*  row_idx){
		auto current_buffer=(DataTypeVect*)&this->reduce_data_buffer[OFFSET];

		auto sm_value_vector=(DataTypeVect*)sm_value;
		auto sm_value_idx_vector=(DataIndexTypeVect*)sm_value_idx;

		constexpr int TILE_NUM= K==1024 ? 2: 1;
		constexpr int BLOCK_SIZE= K==1024 ? 512: K;

		for(int i=0;i<TILE_NUM;i++){
			int raw_row_idx=row_idx[i]*Reduce;
				
			sm_value_vector[threadIdx.x+i*BLOCK_SIZE]=current_buffer[row_idx[i]];

			auto tmp_idx=(threadIdx.x+i*BLOCK_SIZE)*4;

			// printf("%d:%f %f %f %f\n",threadIdx.x,sm_value[tmp_idx],sm_value[tmp_idx+1],
			// 	sm_value[tmp_idx+2],sm_value[tmp_idx+3]);

			sm_value_idx_vector[threadIdx.x+i*BLOCK_SIZE]=VectorTypeMap<int,Reduce>::make(raw_row_idx);
		}


		Top_K<true>::template BitonicSortSharedTopK<DataType,int,
					false,BLOCK_SIZE,Reduce*TILE_NUM,K,false,Reduce*K>(sm_value,sm_value_idx);

		__syncthreads();
		for(int i=0;i<TILE_NUM;i++){
			row_idx[i]=sm_value_idx[threadIdx.x+i*BLOCK_SIZE];
		}

		__syncthreads();

		TopkSelect<K,DataType,ReduceNums...> next_top_select(this->reduce_data_buffer);
		next_top_select.Run(sm_value,sm_value_idx,row_idx);

	}
	DataType* reduce_data_buffer=nullptr;
};




}

template<int K,typename DataType,typename SelectedIdxType,int... ReduceNumAndOffset>
__device__  void topk_all_select_device_fun(DataType* reduce_data_buffer,
							SelectedIdxType* selected_idx,
				SelectedIdxType* outputIdx){
	constexpr int TILE_NUM= K==1024 ? 2: 1;
	constexpr int BLOCK_SIZE= K==1024 ? 512: K;

	int row_idx[2];
	for(int i=0;i<TILE_NUM;i++)
		row_idx[i]=selected_idx[threadIdx.x+i*BLOCK_SIZE];

	TopkSelect<K,DataType,ReduceNumAndOffset...> select(reduce_data_buffer);

	__shared__ DataType sm_value[K*4];
	__shared__ int sm_value_idx[K*4];

	//printf("%d %d\n",sm_value_idx_vector[threadIdx.x].x,sm_value_idx_vector[threadIdx.x].y);
	// printf("sss%d \n",sm_value_idx[threadIdx.x] );	
	select.Run(sm_value,sm_value_idx,row_idx);



    __syncthreads();
	for(int i=0;i<TILE_NUM;i++){
		// printf("%d:%f\n",threadIdx.x+i*BLOCK_SIZE,sm_value[threadIdx.x+i*BLOCK_SIZE]);

		outputIdx[threadIdx.x+i*BLOCK_SIZE]=sm_value_idx[threadIdx.x+i*BLOCK_SIZE];
	}
}
#define DEBUG 0

#ifdef DEBUG
void PrintfCudaMemory(float* data,int size,int cols=16){
	float* host_data=(float*)malloc(sizeof(float)*size);

	cudaMemcpy(host_data,data,sizeof(float)*size,cudaMemcpyDeviceToHost);

	printf("printf_cudaMemory:\n");
	for(int i=0;i<size;i++){
		if(i%cols==0){
			printf("\n");
		}
		printf("%f,",host_data[i]);

		if(i>1024*16){
			break;	
		}
	}
	printf("\n");

	free(host_data);
}

void PrintfCudaMemory(int* data,int size,int cols=16){
	int* host_data=(int*)malloc(sizeof(int)*size);

	cudaMemcpy(host_data,data,sizeof(int)*size,cudaMemcpyDeviceToHost);

	printf("printf_cudaMemory:\n");
	for(int i=0;i<size;i++){
		if(i%cols==0){
			printf("\n");
		}
		printf("%6d,",host_data[i]);

	}
	printf("\n");

	free(host_data);
}
#endif


template<bool IsSorted,typename DataType,typename DataIdxType,int K=16,int BLOCK_SIZE=128,int TILE_NUM=2>
__global__ void topk(DataType* __restrict__ in, DataIdxType* __restrict__ in_idx, DataType* __restrict__ out,
	DataIdxType* __restrict__ out_idx){
	TopkOperation<IsSorted,DataType,DataIdxType>::template
		 Run<BLOCK_SIZE, TILE_NUM,K>(in,in_idx,out,out_idx);
}

template<bool IsSorted,typename DataType,typename DataIdxType,int K=16,int BLOCK_SIZE=128,int TILE_NUM=2>
__global__ void topk(DataType* __restrict__ in,  DataType* __restrict__ out,
	DataIdxType* __restrict__ out_idx,int valid_num){
	TopkOperation<IsSorted,DataType,DataIdxType>::template
		 Run<BLOCK_SIZE, TILE_NUM,K>(in,out,out_idx,valid_num);
}


template<int K,bool IsSorted,typename DataType,typename DataIdxType>
void topk_lunch(DataType* in,DataIdxType* in_idx,DataType* out,DataIdxType* out_idx,
	int DataSize,int valid_num,cudaStream_t& stream){
	// printf("staring topk_lunch:%d\n",DataSize);

	if(DataSize==64){
		constexpr int BLOCK_SIZE=32;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,in_idx,out,out_idx,valid_num);
	}

	if(DataSize==128){
		constexpr int BLOCK_SIZE=64;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,in_idx,out,out_idx,valid_num);
	}

	if(DataSize==256){
		constexpr int BLOCK_SIZE=128;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,in_idx,out,out_idx,valid_num);
	}

	if(DataSize==512){
		constexpr int BLOCK_SIZE=128;
		constexpr int TILE_NUM=4;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,in_idx,out,out_idx,valid_num);
	}

	if(DataSize==1024){
		constexpr int BLOCK_SIZE=512;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,in_idx,out,out_idx,valid_num);
	}

	if(DataSize==2048){
		constexpr int BLOCK_SIZE=512;
		constexpr int TILE_NUM=4;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,in_idx,out,out_idx,valid_num);
	}
	
}

template<int K,bool IsSorted,typename DataType,typename DataIdxType>
void topk_lunch(DataType* in,DataType* out,DataIdxType* out_idx,int DataSize,int valid_num,cudaStream_t& stream){
	// printf("staring topk_lunch:%d\n",DataSize);

	if(DataSize==64){
		constexpr int BLOCK_SIZE=32;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,out,out_idx,valid_num);
	}

	if(DataSize==128){
		constexpr int BLOCK_SIZE=64;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,out,out_idx,valid_num);
	}

	if(DataSize==256){
		constexpr int BLOCK_SIZE=128;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,out,out_idx,valid_num);
	}

	if(DataSize==512){
		constexpr int BLOCK_SIZE=128;
		constexpr int TILE_NUM=4;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,out,out_idx,valid_num);
	}

	if(DataSize==1024){
		constexpr int BLOCK_SIZE=512;
		constexpr int TILE_NUM=2;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,out,out_idx,valid_num);
	}

	if(DataSize==2048){
		constexpr int BLOCK_SIZE=512;
		constexpr int TILE_NUM=4;
		topk<IsSorted,DataType,DataIdxType,K,BLOCK_SIZE,TILE_NUM><<<1,BLOCK_SIZE,0,stream>>>(in,out,out_idx,valid_num);
	}
	
}



#endif

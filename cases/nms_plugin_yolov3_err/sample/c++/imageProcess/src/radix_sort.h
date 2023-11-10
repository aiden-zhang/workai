#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define DEBUG false
#define DEBUG_PERFORMANCE false


namespace dl{

namespace{

template<bool block_size_greater_than_32,int BLOCK_SIZE,typename T>
struct sum_select_by_blocksize
{
	static __device__ auto sum(T value){

		auto c_sum=value;
		constexpr int warp_size=32;
		for(int32_t  stride=1;stride<=(warp_size>>1);stride=stride<<1){
			auto value=__shfl_xor_sync(0xffffffff,c_sum,stride,warp_size);
			c_sum+=value;
		}

		constexpr int NEXT_BLOCK_SIZE=BLOCK_SIZE/warp_size;
		__shared__ T sum_cross_warp[NEXT_BLOCK_SIZE];
		if((threadIdx.x&(warp_size-1))==warp_size-1)
			sum_cross_warp[threadIdx.x>>5]=c_sum;
		__syncthreads();
		auto value_cross_warp=sum_cross_warp[threadIdx.x&(NEXT_BLOCK_SIZE-1)];

		auto value_cross_warp_sum= sum_select_by_blocksize<(NEXT_BLOCK_SIZE>32),NEXT_BLOCK_SIZE,T>::sum(value_cross_warp);
		
		return value_cross_warp_sum;
	}
};
template<int BLOCK_SIZE,typename T>
struct sum_select_by_blocksize<false,BLOCK_SIZE,T>
{
	static __device__ auto sum(T value){

		T c_sum=value;
		for(int32_t  stride=1;stride<=(BLOCK_SIZE>>1);stride=stride<<1){
			auto value=__shfl_xor_sync(0xffffffff,c_sum,stride,BLOCK_SIZE);
			c_sum+=value;
		}
		
		return c_sum;

	}
};

template<typename T,int BLOCK_SIZE>
__device__ auto sum(T value){

	auto cvalue= sum_select_by_blocksize<(BLOCK_SIZE>32),BLOCK_SIZE,T>::sum(value);
	return cvalue;

}

template<bool block_size_greater_than_32,int BLOCK_SIZE,typename T>
struct csum_select_by_blocksize
{
	static __device__ auto csum(T value){

		auto c_sum=value;
		constexpr int warp_size=32;
		for(int32_t  stride=1;stride<=(warp_size>>1);stride=stride<<1){
			auto value=__shfl_up_sync(0xffffffff,c_sum,stride,warp_size);
			if((threadIdx.x&(warp_size-1))>=stride)
				c_sum+=value;
		}

		constexpr int NEXT_BLOCK_SIZE=BLOCK_SIZE/warp_size;
		__shared__ T csum_cross_warp[NEXT_BLOCK_SIZE];
		if((threadIdx.x&(warp_size-1))==warp_size-1)
			csum_cross_warp[threadIdx.x>>5]=c_sum;
		__syncthreads();
		auto value_cross_warp=csum_cross_warp[threadIdx.x&(NEXT_BLOCK_SIZE-1)];

		auto value_cross_warp_csum= csum_select_by_blocksize<(NEXT_BLOCK_SIZE>32),NEXT_BLOCK_SIZE,T>::csum(value_cross_warp);
		
		value=__shfl_sync(0xffffffff,value_cross_warp_csum,(threadIdx.x>>5)-1,NEXT_BLOCK_SIZE);

		if((threadIdx.x>>5) >0){
			c_sum+=value;
		}

		return c_sum;
	}
};
template<int BLOCK_SIZE,typename T>
struct csum_select_by_blocksize<false,BLOCK_SIZE,T>
{
	static __device__ auto csum(T value){

		T c_sum=value;
		for(int32_t  stride=1;stride<=(BLOCK_SIZE>>1);stride=stride<<1){
			auto value=__shfl_up_sync(0xffffffff,c_sum,stride,BLOCK_SIZE);
			if((threadIdx.x&(BLOCK_SIZE-1))>=stride)
				c_sum+=value;
		}
		
		return c_sum;
	}
};

template<typename T,int BLOCK_SIZE>
__device__ auto csum(T value){
	auto cvalue= csum_select_by_blocksize<(BLOCK_SIZE>32),BLOCK_SIZE,T>::csum(value);
	return cvalue;
}
template<typename T,int BLOCK_SIZE,int TILE_NUM>
__device__ auto csum(T* value){
	T psum=0;
	T pcsum[TILE_NUM];


	for(int i=0;i<TILE_NUM;i++){
		psum+=value[i];
		pcsum[i]=i>0 ? value[i]+pcsum[i-1]:value[i];
	}
	auto cvalue= csum_select_by_blocksize<(BLOCK_SIZE>32),BLOCK_SIZE,T>::csum(psum);

	__shared__ T sm[BLOCK_SIZE];
	sm[threadIdx.x]=cvalue;
	__syncthreads();

	
	for(int i=0;i<TILE_NUM;i++){
		value[i]=pcsum[i]+(threadIdx.x>0?sm[threadIdx.x-1]:0);
	}
}

template<int byte_size>
struct ConvertIntoUint
{	
	template<typename T>
	__device__ static auto convert(T& v){
		uint32_t* c=(uint32_t*)&v;
		return *c;
	}
};

template<>
struct ConvertIntoUint<2>
{	
	template<typename T>
	__device__ static auto convert(T& v){
		uint16_t* c=(uint16_t*)&v;
		return *c;
	}
};
template<>
struct ConvertIntoUint<1>
{	
	template<typename T>
	__device__ static auto convert(T& v){
		uint8_t* c=(uint8_t*)&v;
		return *c;
	}
};


template<typename T,bool IsAscending,int BIN_SIZE>
__device__ auto get_bin_id(T value,int BIN_START_IDX){

	T tmp=value;
	auto value_convert=ConvertIntoUint<sizeof(T)>::template convert(tmp);

	auto bin_id=(value_convert>>BIN_START_IDX) & (BIN_SIZE-1);
	if(!IsAscending){
		bin_id=BIN_SIZE-1-bin_id;
	}
	return bin_id;
}


template<int DataSize,int VectorNum>
struct GetVectorMemType
{
	
};

template<>
struct GetVectorMemType<4,4>
{
	typedef int4 DataType;
};

template<>
struct GetVectorMemType<2,4>
{
	typedef int2 DataType;
};

template<>
struct GetVectorMemType<8,4>
{
	typedef double4 DataType;
};


template<int VectorNum,bool Is_Valid_Idxs=true>
struct cond_get_from_vector
{	
	template<typename DataType>
	static __device__ auto get(DataType* value,int32_t idx,
		int32_t valid_num,DataType default_value_tmp=0){
		
		typedef typename GetVectorMemType<sizeof(DataType),VectorNum>::DataType VectorType;

		if(Is_Valid_Idxs==false){
			VectorType c;
			DataType* m=(DataType*)&c;
			#pragma unroll
			for(int i=0;i<VectorNum;i++){
				if(idx*VectorNum+i<valid_num){
					m[i]=idx*VectorNum+i;
				}else{
					m[i]=default_value_tmp;
				}
			}
			return c;
		}else
		if(idx<valid_num/VectorNum){
			return ((VectorType*)value)[idx];
		}else{
			VectorType c;
			DataType* m=(DataType*)&c;
			#pragma unroll
			for(int i=0;i<VectorNum;i++){
				if(idx*VectorNum+i<valid_num){
					m[i]=value[idx*VectorNum+i];
				}else{
					m[i]=default_value_tmp;
				}
			}
			return c;
		}
	}
	template<typename DataType,typename VectorType>
	static __device__ void set(DataType* value,int32_t idx,
		int32_t valid_num,VectorType default_value_tmp){
		
		if(idx<valid_num/VectorNum){
			((VectorType*)value)[idx]=default_value_tmp;
		}else{
			VectorType default_value=default_value_tmp;
			DataType* default_values=(DataType*)&default_value;
			#pragma unroll
			for (int i = 0; i < VectorNum; ++i)
			{
				if(idx*VectorNum+i<valid_num){
					value[idx*VectorNum+i]=default_values[i];
				}
			}
		}
	}
};


template<int tile_id,int end>
struct UnrollLoop
{	
	
	template<typename Func>
	__device__ static  void execute(Func c){
		c(end-tile_id);
				
		UnrollLoop<tile_id-1,end>::template execute(c);

	}

};
template<int end>
struct UnrollLoop<-1,end>
{	
	template<typename Func>
	__device__ static void execute(Func c){
	}
};

template<int start,int end>
struct ForUnroll
{
	template<typename Func>
	__device__ static void execute(Func c){
		constexpr int time=end-start;
		UnrollLoop<time-1,end-1>::template execute(c);
	}
};

template<int tile_id_level0,int end_level0,int tile_id_level1,int end_level1>
struct UnrollLoop2
{	
	template<typename Func0,typename Func1>
	__device__ static  void execute(Func0 c0,Func1 c1){

		auto c_leve0_fun_res=c0(end_level0-tile_id_level0);

		auto c_level1_fun=[&](const int idx){c1(idx,c_leve0_fun_res);};


		UnrollLoop<tile_id_level1,end_level1>::template execute(c_level1_fun);

		UnrollLoop2<tile_id_level0-1,end_level0,tile_id_level1,end_level1>::
			template execute(c0,c1);

	}
};
template<int end_level0,int tile_id_level1,int end_level1>
struct UnrollLoop2<-1,end_level0,tile_id_level1,end_level1>
{	
	template<typename Func0,typename Func1>
	__device__ static  void execute(Func0 c0,Func1 c1){
	}
};


template<int start0,int end0,int start1,int end1>
struct For2Unroll
{
	template<typename Func0,typename Func1>
	__device__ static void execute(Func0 c0,Func1 c1){
		constexpr int time0=end0-start0;
		constexpr int time1=end1-start1;
		UnrollLoop2<time0-1,end0-1,
			time1-1,end1-1>::template execute(c0,c1);
	}
};


template<typename T,bool Open>
struct DataProcessImpl
{	
	template<typename DataType>
	__device__ static auto preprocess(DataType c){
		return c;
	}

	template<typename DataType>
	__device__ static auto postprocess(DataType c){
		return c;
	}
};

template<>
struct DataProcessImpl<float,true>
{
	template<typename DataType>
	__device__ static auto preprocess(DataType& c){

		DataType r;
		uint32_t* c_as_arr=(uint32_t*)&c;
		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>31;

			((uint32_t*)&r)[i]= sign ? ~c_as_arr[i]: (c_as_arr[i] | (1u<<31));
		};

		ForUnroll<0,sizeof(DataType)/sizeof(uint32_t)>::template execute(func);

		return r;
	}

	template<typename DataType>
	__device__ static auto postprocess(DataType& c){
		DataType r;
		uint32_t* c_as_arr=(uint32_t*)&c;
		// int32_t* c_as_arr_int=(int32_t*)&c;
		auto func=[&](const int i){
			

			bool sign=c_as_arr[i]>>31;
			((uint32_t*)&r)[i]= sign ? (c_as_arr[i] & ((1u<<31)-1)) :~c_as_arr[i];
		};

		ForUnroll<0,sizeof(DataType)/sizeof(uint32_t)>::template execute(func);

		return r;
	}
};


template<>
struct DataProcessImpl<__half,true>
{
	template<typename DataType>
	__device__ static auto preprocess(DataType& c){

		DataType r;
		uint16_t* c_as_arr=(uint16_t*)&c;
		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>15;

			((uint16_t*)&r)[i]= sign ? ~c_as_arr[i]: (c_as_arr[i] | (1u<<15));
		};

		ForUnroll<0,sizeof(DataType)/sizeof(uint16_t)>::template execute(func);

		return r;
	}

	template<typename DataType>
	__device__ static auto postprocess(DataType& c){
		DataType r;
		uint16_t* c_as_arr=(uint16_t*)&c;
		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>15;
			((uint16_t*)&r)[i]= sign ? (c_as_arr[i] & ((1u<<15)-1)) :~c_as_arr[i];
		};

		ForUnroll<0,sizeof(DataType)/sizeof(uint16_t)>::template execute(func);

		return r;
	}
};

template<>
struct DataProcessImpl<int32_t,true>
{
	template<typename DataType>
	__device__ static auto preprocess(DataType& c){

		DataType r;
		uint32_t* c_as_arr=(uint32_t*)&c;

		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>31;
			uint32_t mask=~(1u<<31);
			((uint32_t*)&r)[i]= (sign ? (c_as_arr[i] & mask) : (c_as_arr[i] | (mask+1)));
		};

		ForUnroll<0,sizeof(DataType)/sizeof(int32_t)>::template execute(func);

		return r;
	}

	template<typename DataType>
	__device__ static auto postprocess(DataType& c){
		DataType r;
		uint32_t* c_as_arr=(uint32_t*)&c;
		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>31;
			uint32_t mask=~(1u<<31);
			((uint32_t*)&r)[i]= (sign ? (c_as_arr[i]  & mask) : (c_as_arr[i] | (mask+1)));

		};

		ForUnroll<0,sizeof(DataType)/sizeof(int32_t)>::template execute(func);

		return r;
	}
};

template<>
struct DataProcessImpl<int16_t,true>
{
	template<typename DataType>
	__device__ static  auto preprocess(DataType& c){

		DataType r;
		uint16_t* c_as_arr=(uint16_t*)&c;
		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>15;
			uint16_t mask=(1u<<15)-1;
			((uint16_t*)&r)[i]= (sign ? (c_as_arr[i]  & mask) : (c_as_arr[i] | (mask+1)));

		};
		ForUnroll<0,sizeof(DataType)/sizeof(int16_t)>::template execute(func);

		return r;
	}

	template<typename DataType>
	__device__ static  auto postprocess(DataType& c){
		DataType r;
		uint16_t* c_as_arr=(uint16_t*)&c;
		auto func=[&](const int i){
			bool sign=c_as_arr[i]>>15;
			uint16_t mask=(1u<<15)-1;
			((uint16_t*)&r)[i]= (sign ? (c_as_arr[i]  & mask) : (c_as_arr[i] | (mask+1)));
		};
		
		ForUnroll<0,sizeof(DataType)/sizeof(int16_t)>::template execute(func);

		return r;
	}
};


}

template<typename DataType,
        bool IsAscending,int BLOCK_SIZE,int TILE_NUM,int BIN_SIZE,
        typename BIN_COUNTER_DATATYPE=int32_t,
        int VectorNum=4,bool PreProcess=false>

__global__ void radix_sort_bincount(DataType* __restrict__ eles,
	BIN_COUNTER_DATATYPE* __restrict__ bin_count,const int valid_num,int BIN_START_IDX){ //bin_count:[BIN_SIZE][gridDim.x]
	__shared__ int32_t bin_count_sm[BIN_SIZE];
	typedef typename GetVectorMemType<sizeof(DataType),VectorNum>::DataType DataVectorType;
	for(int tidx=threadIdx.x;tidx<BIN_SIZE;tidx+=BLOCK_SIZE){
		bin_count_sm[tidx]=0;
	}
	__syncthreads();
	struct Body0_Result
	{
		DataVectorType value;
		int32_t tidx;	
	};	
	auto func0=[&](const int loop_id){
		int32_t tidx=threadIdx.x+loop_id*BLOCK_SIZE+TILE_NUM/VectorNum*BLOCK_SIZE*blockIdx.x;
		auto value= cond_get_from_vector<VectorNum>::template get(eles,tidx,valid_num,(DataType)0);
		auto value_preprocess=DataProcessImpl<DataType,PreProcess>::template preprocess(value);

		return Body0_Result{value_preprocess,tidx};
	};

	auto func1=[&](const int loop_id,Body0_Result body0_result){
		DataType* values=(DataType*)&(body0_result.value);
		if(body0_result.tidx*VectorNum+loop_id<valid_num){
			auto value_tmp=values[loop_id];
			auto bin_id=get_bin_id<DataType,IsAscending,BIN_SIZE>(value_tmp,BIN_START_IDX);
			atomicAdd(&bin_count_sm[bin_id],1);
		}
	};
	For2Unroll<0,(TILE_NUM/VectorNum),0,VectorNum>::template execute(func0,func1);
	__syncthreads();
	for(int tidx=threadIdx.x;tidx<BIN_SIZE;tidx+=BLOCK_SIZE){
		bin_count[tidx*gridDim.x+blockIdx.x]=bin_count_sm[tidx];
	}
}


template<typename BIN_COUNTER_DATATYPE=int32_t,int CSUM_BLOCK_SIZE=256,int VectorNum=4>
__global__ void radix_sort_csum(BIN_COUNTER_DATATYPE* __restrict__ bin_count,
								BIN_COUNTER_DATATYPE* __restrict__ bin_count_address,
								const int valid_num){ //[BIN_SIZE][gridDim.x]
	
	for(int32_t tid=0;tid<(valid_num+CSUM_BLOCK_SIZE*VectorNum-1)/(CSUM_BLOCK_SIZE*VectorNum);tid++){
		int32_t tidx=tid*CSUM_BLOCK_SIZE+threadIdx.x;
		auto values_tmp=cond_get_from_vector<VectorNum>::template get(bin_count,tidx,valid_num,0);
		BIN_COUNTER_DATATYPE* values=(BIN_COUNTER_DATATYPE*)&values_tmp;
		csum<BIN_COUNTER_DATATYPE,CSUM_BLOCK_SIZE,VectorNum>(values);
		if(tid>0){
			for (int i = 0; i < VectorNum; ++i)
			{
				values[i]+=bin_count_address[tid*CSUM_BLOCK_SIZE*VectorNum-1];
			}
		}	
		cond_get_from_vector<VectorNum>::template set(bin_count_address,tidx,valid_num,values_tmp);
	}
	
}

template<int32_t a,int32_t b>
constexpr int max_value(){
	return a>b?a:b;
}

template<typename DataType,typename IdxType,
        bool IsAscending,int BLOCK_SIZE,int TILE_NUM,int BIN_SIZE,int VectorNum=4,
        typename AddressType=int32_t,bool Is_Valid_Idxs=true,bool PostProcess=false,
        bool PreProcess=false>
__device__ void raidx_sort_reorder_device(DataType*  eles,IdxType*  idxs_dev,
	AddressType*  bin_count_address,
	DataType*  eles_out,IdxType*  idxs_out,
	int valid_num,int BIN_START_IDX=0
	#if DEBUG
	,void*  DebugMem=nullptr
	#endif
	){

	typedef int32_t BIN_COUNTER_TYPE;
	typedef union _bin_counter_tab{
		uint16_t sm_bin_counters[BIN_SIZE/2+1][BLOCK_SIZE][2];
		uint32_t sm_bin_counters_rev[BLOCK_SIZE][BIN_SIZE/2+1];
		uint16_t sm_bin_counters_flaten[(BIN_SIZE/2+1)*BLOCK_SIZE][2];
	}BIN_COUNTER_TAB;

	__shared__ DataType cache_data[TILE_NUM*BLOCK_SIZE];
	__shared__ IdxType cache_idx[TILE_NUM*BLOCK_SIZE];

	__shared__ BIN_COUNTER_TAB bin_counter_tab;

	for(int i=0;i<BIN_SIZE/2+1;i++){
		bin_counter_tab.sm_bin_counters_rev[threadIdx.x][i]=0;
	}
	__syncthreads();
	DataType values[TILE_NUM]={0};
	IdxType idxs[TILE_NUM]={0};
	BIN_COUNTER_TYPE values_idx[TILE_NUM]={0};

	typedef typename GetVectorMemType<sizeof(DataType),VectorNum>::DataType DataVectorType;
	typedef typename GetVectorMemType<sizeof(IdxType),VectorNum>::DataType IndxVectorType;

	struct Body0_Result
	{
		int32_t tidx;
		DataVectorType value;
		IndxVectorType idx;
		int32_t tile_id;
	};
	auto loop0=[&](const int tile_id){
		int32_t tidx=threadIdx.x+TILE_NUM/VectorNum*BLOCK_SIZE*blockIdx.x+tile_id*BLOCK_SIZE;
		auto value_tmp=cond_get_from_vector<VectorNum>::template get(eles,tidx,valid_num,(DataType)0);
		auto value_preprocess=DataProcessImpl<DataType,PreProcess>::template preprocess(value_tmp);

		auto idx_tmp=cond_get_from_vector<VectorNum,Is_Valid_Idxs>::template get(idxs_dev,tidx,valid_num,(IdxType)0);
		return Body0_Result{tidx,value_preprocess,idx_tmp,tile_id};
	};

	auto loop1=[&](const int vid,Body0_Result body0_result){
		DataType* value_tmps=(DataType*)&(body0_result.value);
		IdxType* idx_tmps=(IdxType*)&(body0_result.idx);

		auto now_tile_id=body0_result.tile_id*VectorNum+vid;
		if(body0_result.tidx*VectorNum+vid<valid_num){
			values[now_tile_id]=value_tmps[vid];
			idxs[now_tile_id]=idx_tmps[vid];

			auto bin_id=get_bin_id<DataType,IsAscending,BIN_SIZE>(values[now_tile_id],BIN_START_IDX);

			auto low_id=bin_id&((BIN_SIZE-1)>>1);
			auto high_id=bin_id/(BIN_SIZE>>1);
			uint16_t* counter=&(bin_counter_tab.sm_bin_counters[low_id][threadIdx.x][high_id]);

			values_idx[now_tile_id]=*(counter);
			*counter=*counter+1;
		}else{
			values[now_tile_id]=IsAscending? BIN_SIZE-1:0;
		}
	};
	For2Unroll<0,TILE_NUM/VectorNum,0,VectorNum>::execute(loop0,loop1);
	__syncthreads();

	BIN_COUNTER_TYPE pre_value_sum=0;
	#pragma unroll
	for(int i=0;i<BIN_SIZE/2+1;i++){
		if(threadIdx.x>=1){
			pre_value_sum+=bin_counter_tab.sm_bin_counters_rev[threadIdx.x-1][i];
		}else{
			pre_value_sum+=0;
		}
	}
	__syncthreads();

	auto middle_csum_value=csum<BIN_COUNTER_TYPE,BLOCK_SIZE>(pre_value_sum);

	__syncthreads();
	

	pre_value_sum=middle_csum_value;
	#pragma unroll
	for(int i=0;i<BIN_SIZE/2+1;i++){
		pre_value_sum=bin_counter_tab.sm_bin_counters_rev[threadIdx.x][i]+pre_value_sum;
		bin_counter_tab.sm_bin_counters_rev[threadIdx.x][i]=pre_value_sum;
	}

	__syncthreads();
	//update hight_id=1
	for(int i=0;i<BIN_SIZE/2+1;i++){
		bin_counter_tab.sm_bin_counters[i][threadIdx.x][1]+=
		bin_counter_tab.sm_bin_counters[BIN_SIZE/2][BLOCK_SIZE-1][0];
	}
	__syncthreads();
	auto func_fix_rankid=[&](const int32_t tile_id){
		int32_t tidx=threadIdx.x*VectorNum+tile_id%VectorNum+TILE_NUM*BLOCK_SIZE*blockIdx.x;
		auto bin_id=get_bin_id<DataType,IsAscending,BIN_SIZE>(values[tile_id],BIN_START_IDX);

		auto low_id=bin_id&((BIN_SIZE-1)>>1);
		auto high_id=bin_id/(BIN_SIZE>>1);

		uint16_t prev_counter;
		if ((low_id*BLOCK_SIZE)+threadIdx.x >= 1){
			prev_counter = bin_counter_tab.sm_bin_counters_flaten[(low_id*BLOCK_SIZE)+threadIdx.x-1][high_id];
		}else if(high_id == 0){
			prev_counter = 0;
		} else{
		    prev_counter = bin_counter_tab.sm_bin_counters_flaten[BIN_SIZE/2*BLOCK_SIZE-1][high_id-1];
		}
		auto rank_id=prev_counter+values_idx[tile_id];

		if(tidx<valid_num){ //todo handle rand write
			cache_data[rank_id]=values[tile_id];
			cache_idx[rank_id]=idxs[tile_id];

			
			#if DEBUG
			if(threadIdx.x+tile_id*BLOCK_SIZE<1024 && blockIdx.x==0){
				((DataType*)DebugMem)[threadIdx.x+tile_id*BLOCK_SIZE]=values[tile_id];
			}
			#endif
		}
	};

	ForUnroll<0,TILE_NUM>::template execute(func_fix_rankid);
	
	__syncthreads();

	auto func_write=[&](const int32_t i){
		int rank_id=threadIdx.x+i*BLOCK_SIZE;
		auto value=cache_data[rank_id];
		auto idx=cache_idx[rank_id];

		auto bin_id=get_bin_id<DataType,IsAscending,BIN_SIZE>(value,BIN_START_IDX);

		auto prev_low_id=(bin_id-1)&((BIN_SIZE-1)>>1);
		auto prev_high_id=(bin_id-1)/(BIN_SIZE>>1);

		auto bin_rank=(bin_id>=1 ?(rank_id-bin_counter_tab.sm_bin_counters[prev_low_id][BLOCK_SIZE-1][prev_high_id]) 
			:rank_id);

		auto bin_offset=blockIdx.x+bin_id*gridDim.x > 0 ? bin_count_address[blockIdx.x+bin_id*gridDim.x-1]:0;
		int32_t output_address=bin_offset+bin_rank;

		if(TILE_NUM*BLOCK_SIZE*blockIdx.x+rank_id<valid_num){

			auto value_postprocess=DataProcessImpl<DataType,PostProcess>::template postprocess(value);

			eles_out[output_address]=value_postprocess;
			idxs_out[output_address]=idx;
		}
	};
	ForUnroll<0,TILE_NUM>::template execute(func_write);
}


template<typename DataType,typename IdxType,
        bool IsAscending,int BLOCK_SIZE,int TILE_NUM,int BIN_SIZE,int VectorNum=4,
        typename AddressType=int32_t,bool PostProcess=false,bool PreProcess=false>
__global__ void raidx_sort_reorder(DataType* __restrict__  eles,IdxType* __restrict__ idxs_dev,
	AddressType* __restrict__ bin_count_address,
	DataType* __restrict__ eles_out,IdxType* __restrict__ idxs_out,
	int valid_num,int BIN_START_IDX=0
	#if DEBUG
	,void* __restrict__ DebugMem=nullptr
	#endif
	){

	raidx_sort_reorder_device<DataType,IdxType,IsAscending,
		BLOCK_SIZE,TILE_NUM,BIN_SIZE,VectorNum,AddressType,true,PostProcess,PreProcess>(eles,idxs_dev,bin_count_address,
			eles_out,idxs_out,valid_num,BIN_START_IDX
			#if DEBUG
			, DebugMem
			#endif
			);
}

template<typename DataType,typename IdxType,
        bool IsAscending,int BLOCK_SIZE,int TILE_NUM,int BIN_SIZE,int VectorNum=4,
        typename AddressType=int32_t,bool PostProcess=false,bool PreProcess=false>
__global__ void raidx_sort_reorder(DataType* __restrict__  eles,
	AddressType* __restrict__ bin_count_address,
	DataType* __restrict__ eles_out,IdxType* __restrict__ idxs_out,
	int valid_num,int BIN_START_IDX=0
	#if DEBUG
	,void* __restrict__ DebugMem=nullptr
	#endif
	){

	raidx_sort_reorder_device<DataType,IdxType,IsAscending,
		BLOCK_SIZE,TILE_NUM,BIN_SIZE,VectorNum,AddressType,false,PostProcess,PreProcess>(eles,nullptr,bin_count_address,
			eles_out,idxs_out,valid_num,BIN_START_IDX
			#if DEBUG
			, DebugMem
			#endif
			);
}


namespace{

template<typename DataType,typename IndxType,
			bool IsAscending,
			int BLOCK_SIZE=256,int TILE_NUM=8>
struct RadixSort
{	

	template<typename BIN_COUNTER_DATATYPE, bool PreProcess=false>
	static void bin_count(dim3 grid,dim3 block,
		int32_t current_bin_size,
		DataType* data_dev,BIN_COUNTER_DATATYPE* bin_count_dev,
		int total_num,
		int BIN_START_IDX,cudaStream_t* stream){
		switch(current_bin_size){
			case 4: dl::radix_sort_bincount<DataType,IsAscending,BLOCK_SIZE,TILE_NUM,4,BIN_COUNTER_DATATYPE,4,PreProcess>
			    			<<<grid,block,0,*stream>>>(data_dev,bin_count_dev,total_num,BIN_START_IDX);break;

		    case 16: dl::radix_sort_bincount<DataType,IsAscending,BLOCK_SIZE,TILE_NUM,16,BIN_COUNTER_DATATYPE,4,PreProcess>
			    			<<<grid,block,0,*stream>>>(data_dev,bin_count_dev,total_num,BIN_START_IDX);break;

			case 32: dl::radix_sort_bincount<DataType,IsAscending,BLOCK_SIZE,TILE_NUM,32,BIN_COUNTER_DATATYPE,4,PreProcess>
			    			<<<grid,block,0,*stream>>>(data_dev,bin_count_dev,total_num,BIN_START_IDX);break;

			case 64: dl::radix_sort_bincount<DataType,IsAscending,BLOCK_SIZE,TILE_NUM,64,BIN_COUNTER_DATATYPE,4,PreProcess>
			    			<<<grid,block,0,*stream>>>(data_dev,bin_count_dev,total_num,BIN_START_IDX);break;
		}

	}
	template<typename BIN_COUNTER_DATATYPE>
	static void csum(int32_t current_bin_size,
			BIN_COUNTER_DATATYPE* bin_count_dev,
			BIN_COUNTER_DATATYPE* bin_count_address,
			int block_num,
			cudaStream_t* stream ){
			
		switch(sizeof(BIN_COUNTER_DATATYPE))
		{
			case 4: dl::radix_sort_csum<BIN_COUNTER_DATATYPE,512,4>
		    		<<<dim3(1,1,1),dim3(512,1,1),0,*stream>>>
		    (bin_count_dev,bin_count_address,block_num*current_bin_size);break;
		}	   
	}

	template<typename BIN_COUNTER_DATATYPE,bool PostProcess=false,bool PreProcess=false>
	static void reorder_without_idxs(dim3 grid,dim3 block,
			int32_t current_bin_size,
			DataType* data_dev,IndxType* indxs_dev,BIN_COUNTER_DATATYPE* bin_count_address,
			DataType* data_dev_swap,IndxType* idxs_dev_swap,int total_num,int BIN_START_IDX,cudaStream_t* stream
			#if DEBUG
			,void* DebugMem=nullptr
			#endif
			){

		        switch(current_bin_size){
					case 4: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
								4,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;

			    	case 16: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
			    	            16,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;
				    case 32: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
				                 32,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;

				    case 64: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
				             64,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;
			    }

	}
	template<typename BIN_COUNTER_DATATYPE,bool PostProcess=false,bool PreProcess=false>
	static void reorder_with_idxs(dim3 grid,dim3 block,
			int32_t current_bin_size,
			DataType* data_dev,IndxType* indxs_dev,BIN_COUNTER_DATATYPE* bin_count_address,
			DataType* data_dev_swap,IndxType* idxs_dev_swap,int total_num,int BIN_START_IDX,cudaStream_t* stream
			#if DEBUG
			,void* DebugMem=nullptr
			#endif
			){

			switch(current_bin_size){
					case 4: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
							4,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,indxs_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;

			    	case 16: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
			    	         16,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,indxs_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;
				    case 32: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
				             32,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,indxs_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;

				    case 64: dl::raidx_sort_reorder<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM,
				             64,4,BIN_COUNTER_DATATYPE,PostProcess,PreProcess>
				    	<<<grid,block,0,*stream>>>
				    		(data_dev,indxs_dev,bin_count_address,
				    			data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX
				    			#if DEBUG
				    			,DebugMem
				    			#endif
				    			);break;
			    }

	}
	template<typename BIN_COUNTER_DATATYPE,bool PostProcess=false,bool PreProcess=false>
	static void reorder(dim3 grid,dim3 block,
			int32_t current_bin_size,
			DataType* data_dev,IndxType* indxs_dev,BIN_COUNTER_DATATYPE* bin_count_address,
			DataType* data_dev_swap,IndxType* idxs_dev_swap,int total_num,int BIN_START_IDX,cudaStream_t* stream
			#if DEBUG
			,void* DebugMem=nullptr
			#endif
		){

		if(indxs_dev==nullptr){
			RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template
			 reorder_without_idxs<BIN_COUNTER_DATATYPE,PostProcess,PreProcess>(grid,block,current_bin_size,
				data_dev,indxs_dev,bin_count_address,data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX,stream);
		}else{
			RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template
			  reorder_with_idxs<BIN_COUNTER_DATATYPE,PostProcess,PreProcess>(grid,block,current_bin_size,
				data_dev,indxs_dev,bin_count_address,data_dev_swap,idxs_dev_swap,total_num,BIN_START_IDX,stream);

			}
	}
};

template<typename T>
void swap_pointer(T& a,T& b){
  T tmp=a;
  a=b;
  b=tmp;
}
#if DEBUG_PERFORMANCE
#define ProfilerStart(NAME, stream,Rep)  \
 do{                                \
 	using namespace std;            \
 	cudaEvent_t start,stop;         \
 	std::string kernel_name= NAME;  \
 	int rep= Rep;                   \
 	cudaEventCreate(&start);        \
	cudaEventCreate(&stop);         \
	cudaEventSynchronize(start);    \
	cudaEventRecord(start,stream);  \
	for (int i = 0; i < rep ; ++i)  \
	{

#define ProfilerEnd(stream)         \
	}		    	                \
	cudaEventRecord(stop,stream);   \
	cudaEventSynchronize(stop);     \
	float time_elapsed;             \
	cudaEventElapsedTime(&time_elapsed,start,stop);     \
	cudaEventDestroy(start);                            \
	cudaEventDestroy(stop);                             \
	std::cout<<"=============================="<<std::endl;    \
	std::cout<<"****"<<kernel_name<<"******\ntotal_time:"<<time_elapsed<<"(ms)\t"<< \
	" "<<time_elapsed*1000/rep<<"(avg_us)\n=============================="<<std::endl; \
}while(0)
#else
#define ProfilerStart(NAME, stream,Rep){};
#define ProfilerEnd(stream){};
#endif
template<typename T>
void printf_cuda_memory(T * value,int size,std::string name="value",int row_break=32){
	cudaDeviceSynchronize();
	T* cpu_v=(T*)malloc(sizeof(T)*size);

	cudaMemcpy(cpu_v,value,sizeof(T)*size,cudaMemcpyDeviceToHost);
	
	std::stringstream ss;
	ss<<name<<":\n[";
	for (int i = 0; i < size; ++i)
	{
		ss<<cpu_v[i]<<",";
		if((i+1)%row_break==0){
			ss<<std::endl;
		}
	}
	ss<<"]"<<std::endl;

	std::cout<<ss.str();
}
}


//indxs_dev maybe is nullptr
template<typename DataType,typename IndxType,bool IsAscending=false>
int32_t RadixSortFunc(DataType* data_dev,IndxType* indxs_dev,int32_t total_num,
	DataType* data_out_dev=nullptr,
	IndxType* indxs_out_dev=nullptr,
	void* workspace_dev=nullptr,
	cudaStream_t* stream=nullptr){
	//when workspace_dev is nullptr return workspace size

	constexpr int BLOCK_SIZE=256;
	constexpr int TILE_NUM=4;
	constexpr int MAX_BIN_SIZE=64;

	int block_num=(total_num+BLOCK_SIZE*TILE_NUM-1)/(BLOCK_SIZE*TILE_NUM);


	// int vector_num=16/(sizeof(DataType)>sizeof(IndxType)?sizeof(DataType):sizeof(IndxType));

	// int32_t align_total_num=(total_num+vector_num-1)/vector_num *vector_num;
	
	int32_t data_align= (sizeof(DataType)*total_num)+(16-(sizeof(DataType)*total_num)%16);
	int32_t indx_align= (data_align+sizeof(IndxType)*total_num)+(16-(data_align+sizeof(IndxType)*total_num)%16);

	if(workspace_dev==nullptr){

		return indx_align+block_num*MAX_BIN_SIZE*sizeof(int32_t)*2+total_num*sizeof(int32_t);

	}else{
		dim3 grid(block_num,1,1);
	    dim3 block(BLOCK_SIZE,1,1);

	    DataType* data_dev_swap=(DataType*)workspace_dev;
	    IndxType* indx_dev_swap=(IndxType*)(((char*)workspace_dev)+data_align);

	    int32_t* bin_count_dev=(int32_t*)(((char*)workspace_dev)+indx_align);
	  	int32_t* bin_adddress_dev=bin_count_dev+block_num*MAX_BIN_SIZE;
	    // int current_log2_bin_size=4;
	    // int adjust_log2_bin_size_idx=sizeof(DataType)==4? 8:4;
	    int current_log2_bin_size=5;
	    int adjust_log2_bin_size_idx=sizeof(DataType)==4? 20:10;
	    
	    #if DEBUG
	    uint32_t* DebugMem;
	    cudaMalloc((void**)&DebugMem,1024*sizeof(DataType));
	    cudaMemset(DebugMem,0,1024*sizeof(DataType));
	    #endif

	    if(sizeof(DataType)==2){
	    	swap_pointer(data_out_dev,data_dev_swap);
		 	swap_pointer(indxs_out_dev,indx_dev_swap);
	    }

	    for(int BIN_START_IDX=0;BIN_START_IDX<sizeof(DataType)*8;
	    	BIN_START_IDX+=current_log2_bin_size){

	    	#if DEBUG 
	    	printf("BIN_START_IDX: %d\n",BIN_START_IDX );
	    	#endif

	    	if(BIN_START_IDX==adjust_log2_bin_size_idx){
	    		current_log2_bin_size=6;
	    	}
	    	int32_t current_bin_size= 1<<current_log2_bin_size;

	    	// printf("bin_count:%d\n",current_bin_size);
	    	ProfilerStart("bin_count",*stream,10);

	    	if(BIN_START_IDX==0){
	    		RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template bin_count<int32_t,true>(
	    			grid,block,current_bin_size,
		    		data_dev,bin_count_dev,total_num,BIN_START_IDX,stream);
	    	}else{
	    		RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template bin_count<int32_t,false>(
	    			grid,block,current_bin_size,
		    		data_dev,bin_count_dev,total_num,BIN_START_IDX,stream);
	    	}

	    	ProfilerEnd(*stream);
			#if DEBUG
	    	printf_cuda_memory(bin_count_dev,current_bin_size*grid.x,"bin_count_dev");
	    	#endif

		    ProfilerStart("csum",*stream,10);
		    RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template csum<int32_t>(
		    	current_bin_size,
		    	bin_count_dev,bin_adddress_dev,block_num,stream);
		    ProfilerEnd(*stream);
		    #if DEBUG
		    printf_cuda_memory(bin_adddress_dev,current_bin_size*grid.x,"bin_adddress_dev");
		    #endif

			// printf("reorder\n" );
			ProfilerStart("reorder",*stream,10);

			if(BIN_START_IDX+current_log2_bin_size>=sizeof(DataType)*8){
				RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template reorder<int32_t,true,false>(
			    	grid,block,current_bin_size,
			    	data_dev,indxs_dev,bin_adddress_dev,
				    data_dev_swap,indx_dev_swap,total_num,BIN_START_IDX,stream
				    #if DEBUG
				    ,DebugMem
				    #endif
				    );
			}else{
				if(BIN_START_IDX==0){
		    		RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template reorder<int32_t,false,true>(
				    	grid,block,current_bin_size,
				    	data_dev,indxs_dev,bin_adddress_dev,
					    data_dev_swap,indx_dev_swap,total_num,BIN_START_IDX,stream
					    #if DEBUG
					    ,DebugMem
					    #endif
					    );
		    	}else{

		    		RadixSort<DataType,IndxType,IsAscending,BLOCK_SIZE,TILE_NUM>::template reorder<int32_t,false,false>(
				    	grid,block,current_bin_size,
				    	data_dev,indxs_dev,bin_adddress_dev,
					    data_dev_swap,indx_dev_swap,total_num,BIN_START_IDX,stream
					    #if DEBUG
					    ,DebugMem
					    #endif
					    );
		    	}
			}
		    ProfilerEnd(*stream);
		  
		    #if DEBUG
		    printf_cuda_memory(data_dev_swap,total_num,"data_dev_swap");
		    printf_cuda_memory(indx_dev_swap,total_num,"indx_dev_swap");

		    printf_cuda_memory(DebugMem,1024,"tmp_value");
		    #endif

		 	if(BIN_START_IDX==0){
		 		swap_pointer(data_dev,data_out_dev);
		 		swap_pointer(indxs_dev,indxs_out_dev);	
		 	}
		 	swap_pointer(data_dev,data_dev_swap);
		 	swap_pointer(indxs_dev,indx_dev_swap);
            
	    }
		return 0;
	}
}	
}

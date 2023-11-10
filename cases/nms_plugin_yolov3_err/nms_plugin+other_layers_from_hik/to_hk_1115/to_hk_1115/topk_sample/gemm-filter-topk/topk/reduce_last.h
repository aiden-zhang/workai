#ifndef REDUCE_LAST_H_
#define REDUCE_LAST_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include<cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <vector>

using namespace std;

#include "./topk.h"


struct float32x8
{
	float4 x;
	float4 y;
};


template<typename T>
struct VALUE
{
};

__device__ float bitcast(uint32_t a){
	return *((float*)&a);
}
__device__ half bitcast(uint16_t a){
	return *((half*)&a);
}

template<>
struct VALUE<float>
{
	__device__ static float min(){
	 	return bitcast((((1u<<31)-1) xor (1u<<23)) | (1u<<31));
	 }
	__device__ static float max(){
	 	return bitcast((((1u<<31)-1) xor (1u<<23)));
	 }
};

template<typename DataType,int NUM>
struct VectorDataType
{
	
};

template<>
struct VectorDataType<float,4>
{
	typedef float4 DataType;
};

template<>
struct VectorDataType<float,2>
{
	typedef float2 DataType;
};

template<>
struct VectorDataType<half,4>
{
	typedef float2 DataType;
};
template<>
struct VectorDataType<half,8>
{
	typedef float4 DataType;
};

template<>
struct VectorDataType<half,2>
{
	typedef half2 DataType;
};


__device__ float max_2(float a,float b){
	return ::fmax(a,b);
}

__device__ float max_4(float a,float b,float c,float d){
	return max_2(max_2(a,b),max_2(c,d));
}

__device__ float max(float4 a){
	return max_4(a.x,a.y,a.z,a.w);
}
__device__ float max(float2 a){
	return max_2(a.x,a.y);
}

__device__ float max(float32x8 a){
	return max_2(max(a.x),max(a.y));
}

template<typename DataType,int... ReduceNumsAndOffset>
class ReduceLastAll
{
public:

	__device__ ReduceLastAll(DataType* out,int tile_idx,int tile_num){
		
	}
	__device__ void Run(DataType current_max_value,int idx,int csum_reduce_num,int WrapSize=2){


	}
	DataType* out=nullptr;
};

template<typename DataType,int ReduceNum,int Offset,int... ReduceNumsAndOffset>
class ReduceLastAll<DataType,ReduceNum,Offset,ReduceNumsAndOffset...>
{
public:
	typedef typename VectorDataType<DataType,ReduceNum>::DataType DataTypeVector;

	__device__ ReduceLastAll(DataType* out,int tile_idx,int tile_num){
		this->out=out;
		this->tile_idx=tile_idx;
		this->tile_num=tile_num;
	}
	__device__ void Run(DataType current_max_value,int idx,int csum_reduce_num=1,int WrapSize=2){

		auto back_value=__shfl_down_sync(0xFFFFFFFF,current_max_value,WrapSize>>1,WrapSize);

		auto next_current_max_value=max_2(back_value,current_max_value);

		if(ReduceNum==4){
			WrapSize=WrapSize*2;
			back_value=__shfl_down_sync(0xFFFFFFFF,next_current_max_value,WrapSize>>1,WrapSize);
			next_current_max_value=max_2(back_value,next_current_max_value);
		}

		csum_reduce_num=csum_reduce_num*ReduceNum;

		if((idx +tile_idx*this->tile_num)%csum_reduce_num==0){
			this->out[(idx +tile_idx*this->tile_num)/csum_reduce_num+Offset]=next_current_max_value;
		}

		ReduceLastAll<DataType,ReduceNumsAndOffset...> next_reduce_last(this->out,tile_idx,this->tile_num);
		
		next_reduce_last.Run(next_current_max_value,idx,csum_reduce_num,WrapSize<<1);

	}
	DataType* out=nullptr;
	int tile_idx;
	int tile_num;
};


template<int ReduceNum,typename DataType>
struct CondVectorGet{

	__device__ static float4 Get(float* buffer,int idx,int valid_num){
		float4 in4;
		in4.x= idx*ReduceNum+0 <valid_num? buffer[idx*ReduceNum+0] : VALUE<DataType>::min();
		in4.y= idx*ReduceNum+1 <valid_num? buffer[idx*ReduceNum+1] : VALUE<DataType>::min();
		in4.z= idx*ReduceNum+2 <valid_num? buffer[idx*ReduceNum+2] : VALUE<DataType>::min();
		in4.w= idx*ReduceNum+3 <valid_num? buffer[idx*ReduceNum+3] : VALUE<DataType>::min();
		return in4;
	}
};

template<>
struct CondVectorGet<2,float>{

	__device__ static float2 Get(float* buffer,int idx,int valid_num){
		float2 in2;
		in2.x= idx*2+0 <valid_num? buffer[idx*2+0] : VALUE<float>::min();
		in2.y= idx*2+1 <valid_num? buffer[idx*2+1] : VALUE<float>::min();
		
		return in2;
	}
};

template<>
struct CondVectorGet<2,half>{

	__device__ static float2 Get(half* buffer,int idx,int valid_num){
		float2 in2;
		in2.x= idx*2+0 <valid_num? float(buffer[idx*2+0]) : VALUE<float>::min();
		in2.y= idx*2+1 <valid_num? float(buffer[idx*2+1]) : VALUE<float>::min();
		
		return in2;
	}
};

template<>
struct CondVectorGet<4,half>{

	__device__ static float4 Get(half* buffer,int idx,int valid_num){
		float4 in2;
		in2.x= idx*4+0 <valid_num? float(buffer[idx*4+0]) : VALUE<float>::min();
		in2.y= idx*4+1 <valid_num? float(buffer[idx*4+1]) : VALUE<float>::min();
		in2.z= idx*4+2 <valid_num? float(buffer[idx*4+2]) : VALUE<float>::min();
		in2.w= idx*4+3 <valid_num? float(buffer[idx*4+3]) : VALUE<float>::min();
		return in2;
	}
};


template<>
struct CondVectorGet<8,half>{

	__device__ static float32x8 Get(half* buffer,int idx,int valid_num){
		float32x8 in2;
		in2.x.x= idx*8+0 <valid_num? float(buffer[idx*8+0]) : VALUE<float>::min();
		in2.x.y= idx*8+1 <valid_num? float(buffer[idx*8+1]) : VALUE<float>::min();
		in2.x.z= idx*8+2 <valid_num? float(buffer[idx*8+2]) : VALUE<float>::min();
		in2.x.w= idx*8+3 <valid_num? float(buffer[idx*8+3]) : VALUE<float>::min();

		in2.y.x= idx*8+4 <valid_num? float(buffer[idx*8+4]) : VALUE<float>::min();
		in2.y.y= idx*8+5 <valid_num? float(buffer[idx*8+5]) : VALUE<float>::min();
		in2.y.z= idx*8+6 <valid_num? float(buffer[idx*8+6]) : VALUE<float>::min();
		in2.y.w= idx*8+7 <valid_num? float(buffer[idx*8+7]) : VALUE<float>::min();

		return in2;
	}
};

__device__ auto vectorToFloatx8FromHalf8(float4 c){
	float32x8 r;

	float tmp=c.x;
	r.x.x=((half2*)&tmp)->x;
	r.x.y=((half2*)&tmp)->y;

	tmp=c.y;
	r.x.z=((half2*)&tmp)->x;
	r.x.w=((half2*)&tmp)->y;

	tmp=c.z;
	r.y.x=((half2*)&tmp)->x;
	r.y.y=((half2*)&tmp)->y;

	tmp=c.w;
	r.y.z=((half2*)&tmp)->x;
	r.y.w=((half2*)&tmp)->y;

	return r;
}

template<typename DataType,int FirstWrite,int... ReduceNumsAndOffset>
struct ReduceLastImpl{
};
template<int... ReduceNumsAndOffset>
struct ReduceLastImpl<half,0,8,0,ReduceNumsAndOffset...>{

static __device__ void ReduceLastAllDevice(half* in,float* out,int valid_num,int tile_idx,int tile_num){


	int idx=threadIdx.x+blockDim.x*blockIdx.x;

	float4* in_ptr=(float4*)(&in[0]);

	float32x8 in4;
	
	int idx_nums=idx+tile_idx*tile_num;

	
	// if(idx_nums*8+7>=valid_num && valid_num>0){
	// 	in4= CondVectorGet<8,half>::Get(in,idx_nums,valid_num);
	// }else{
	auto half8_val=in_ptr[idx_nums];

	in4 = vectorToFloatx8FromHalf8(half8_val);

	// }
	float current_max_value=max(in4);
	

	ReduceLastAll<float,ReduceNumsAndOffset...> reduce_all(out,tile_idx,tile_num);
	reduce_all.Run(current_max_value,idx,1,2);
}

template<int tile_idx,typename InDataType, typename... InDataTypes>
static __device__ void ReduceLastAllDevice(float* out,int valid_num,int tile_num,
	InDataType in,InDataTypes... ins){

	
	ReduceLastAllDevice(in,out,valid_num,tile_idx,tile_num);

	ReduceLastAllDevice<tile_idx+1,InDataTypes...>(out,valid_num,tile_num,ins...);


}
template<int start_tile_idx>
static __device__ void ReduceLastAllDevice(float* out,int valid_num,int tile_num){

}

};

template<int... ReduceNumsAndOffset>
struct ReduceLastImpl<half,-1,8,0,ReduceNumsAndOffset...>{

static __device__ void ReduceLastAllDevice(half* in,float* out,int valid_num,
		int tile_idx,int tile_num,int offset){


	int idx=threadIdx.x+blockDim.x*blockIdx.x+offset;

	float4* in_ptr=(float4*)(&in[0]);

	float32x8 in4;
	
	int idx_nums=idx+tile_idx*tile_num;

	
	if(idx_nums*8+7>=valid_num && valid_num>0){
		in4= CondVectorGet<8,half>::Get(in,idx_nums,valid_num);
	}else{
		auto half8_val=in_ptr[idx];

		in4 = vectorToFloatx8FromHalf8(half8_val);

	}
	float current_max_value=max(in4);
	
	

	ReduceLastAll<float,ReduceNumsAndOffset...> reduce_all(out,tile_idx,tile_num);
	reduce_all.Run(current_max_value,idx,1,2);
}

template<int tile_idx,typename InDataType, typename... InDataTypes>
static __device__ void ReduceLastAllDevice(float* out,int valid_num,int tile_num,int offset,
	InDataType in,InDataTypes... ins){

	
	ReduceLastAllDevice(in,out,valid_num,tile_idx,tile_num,offset);

	ReduceLastAllDevice<tile_idx+1,InDataTypes...>(out,valid_num,tile_num,offset,ins...);


}
template<int start_tile_idx>
static __device__ void ReduceLastAllDevice(float* out,int valid_num,int offset,int tile_num){

}

};


template<int... ReduceNumsAndOffset>
struct ReduceLastImpl<float,-1,4,0,ReduceNumsAndOffset...>{


static __device__ void ReduceLastAllDevice(float* in,float* out,int valid_num,
	int tile_idx,int tile_num,int offset){
	typedef typename VectorDataType<float,4>::DataType DataTypeVector;

	int idx=threadIdx.x+blockDim.x*blockIdx.x+offset;

	DataTypeVector* in_ptr=(DataTypeVector*)(&in[0]);

	DataTypeVector in4;

	
	int idx_nums=idx+tile_idx*tile_num;

	
	if(idx_nums*4+4-1>=valid_num && valid_num>0){

		in4= CondVectorGet<4,float>::Get(in,idx_nums,valid_num);
	}else{
		in4=in_ptr[idx];
	}
	float current_max_value=max(in4);

	ReduceLastAll<float,ReduceNumsAndOffset...> reduce_all(out,tile_idx,tile_num);
	reduce_all.Run(current_max_value,idx,1,2);
}

template<int tile_idx,typename InDataType, typename... InDataTypes>
static __device__ void ReduceLastAllDevice(float* out,int valid_num,int tile_num,int offset,
	InDataType in,
	InDataTypes... ins){

	
	ReduceLastAllDevice(in,out,valid_num,tile_idx,tile_num,offset);

	ReduceLastAllDevice<tile_idx+1,InDataTypes...>(out,valid_num,tile_num,offset,ins...);


}
template<int start_tile_idx>
static __device__ void ReduceLastAllDevice(float* out,int valid_num,int offset,int tile_num){

}
};

template<typename DataType,int FirstWrite,int ReduceNum,int Offset,int... ReduceNumsAndOffset>
struct ReduceLastImpl<DataType,FirstWrite,ReduceNum,Offset,ReduceNumsAndOffset...>{


static __device__ void ReduceLastAllDevice(DataType* in,DataType* out,int valid_num=0,int tile_idx=0,int tile_num=0){
	typedef typename VectorDataType<DataType,ReduceNum>::DataType DataTypeVector;

	int idx=threadIdx.x+blockDim.x*blockIdx.x;

	DataTypeVector* in_ptr=(DataTypeVector*)(&in[0]);

	DataTypeVector in4;

	
	int idx_nums=idx+tile_idx*tile_num;

	
	if(idx_nums*ReduceNum+ReduceNum-1>=valid_num && valid_num>0){

		in4= CondVectorGet<ReduceNum,DataType>::Get(in,idx_nums,valid_num);
	}else{
		in4=in_ptr[idx];
	}

	
	DataType current_max_value=max(in4);
	
	if(Offset!=0 or FirstWrite)
		out[idx_nums]=current_max_value;

	ReduceLastAll<DataType,ReduceNumsAndOffset...> reduce_all(out,tile_idx,tile_num);
	reduce_all.Run(current_max_value,idx,1,2);
}

template<int tile_idx,typename InDataType, typename... InDataTypes>
static __device__ void ReduceLastAllDevice(DataType* out,int valid_num,int tile_num,
	InDataType in,InDataTypes... ins){

	
	ReduceLastAllDevice(in,out,valid_num,tile_idx,tile_num);

	ReduceLastAllDevice<tile_idx+1,InDataTypes...>(out,valid_num,tile_num,ins...);


}
template<int start_tile_idx>
static __device__ void ReduceLastAllDevice(DataType* out,int valid_num,int tile_num){

}


template<int FIRST_OFFSET>
static __device__ void ReduceLastAllDevice(DataType* out,int valid_num){
	typedef typename VectorDataType<DataType,ReduceNum>::DataType DataTypeVector;

	int idx=threadIdx.x+blockDim.x*blockIdx.x;

	// int FIRST_OFFSET=Offset-blockDim.x*gridDim.x*ReduceNum;
	DataTypeVector* in_ptr=(DataTypeVector*)(&out[FIRST_OFFSET]);
	DataTypeVector in4;

	if(idx*ReduceNum+ReduceNum-1>=valid_num && valid_num>0){
		in4= CondVectorGet<ReduceNum,DataType>::Get((DataType*)in_ptr,idx,valid_num);
	}else{
		in4=in_ptr[idx];
	}
	
	DataType current_max_value=max(in4);
	if(Offset!=0)
		out[Offset+idx]=current_max_value;

	ReduceLastAll<DataType,ReduceNumsAndOffset...> reduce_all(out,0,blockDim.x*gridDim.x);
	reduce_all.Run(current_max_value,idx,1,2);
}
};


template<int FirstWrite,int ReduceNum,int Offset,int... ReduceNumsAndOffset>
struct ReduceLastImpl<half,FirstWrite,ReduceNum,Offset,ReduceNumsAndOffset...>{
typedef float DataType;

static __device__ void ReduceLastAllDevice(half* in,DataType* out,int valid_num=0,
										int tile_idx=0,int tile_num=0){
	typedef typename VectorDataType<half,ReduceNum>::DataType DataTypeVector;

	int idx=threadIdx.x+blockDim.x*blockIdx.x;

	DataTypeVector* in_ptr=(DataTypeVector*)(&in[0]);

	float4 in4;

	
	int idx_nums=idx+tile_idx*tile_num;

	
	if(idx_nums*ReduceNum+ReduceNum-1>=valid_num && valid_num>0){
			in4= CondVectorGet<ReduceNum,half>::Get(in,idx_nums,valid_num);
	}else{
		 	float2 value=in_ptr[idx];
		 	float tmp=value.x;
		 	half2* v_h2=(half2*)&tmp;

		 	in4.x=float(v_h2->x);
		 	in4.y=float(v_h2->y);

		 	tmp=value.y;
		 	v_h2=(half2*)&tmp;

		 	in4.z=float(v_h2->x);
		 	in4.w=float(v_h2->y);
	}


	DataType current_max_value=max(in4);
	

	if(Offset!=0 or FirstWrite)
		out[idx_nums]=current_max_value;

	ReduceLastAll<DataType,ReduceNumsAndOffset...> reduce_all(out,tile_idx,tile_num);
	reduce_all.Run(current_max_value,idx,1,2);
}

template<int tile_idx,typename InDataType, typename... InDataTypes>
static __device__ void ReduceLastAllDevice(DataType* out,int valid_num,int tile_num,
	InDataType in,InDataTypes... ins){

	
	ReduceLastAllDevice(in,out,valid_num,tile_idx,tile_num);

	ReduceLastAllDevice<tile_idx+1,InDataTypes...>(out,valid_num,tile_num,ins...);


}
template<int start_tile_idx>
static __device__ void ReduceLastAllDevice(DataType* out,int valid_num,int tile_num){

}


template<int FIRST_OFFSET>
static __device__ void ReduceLastAllDevice(DataType* out,int valid_num){
	typedef typename VectorDataType<DataType,ReduceNum>::DataType DataTypeVector;

	int idx=threadIdx.x+blockDim.x*blockIdx.x;

	// int FIRST_OFFSET=Offset-blockDim.x*gridDim.x*ReduceNum;
	DataTypeVector* in_ptr=(DataTypeVector*)(&out[FIRST_OFFSET]);
	DataTypeVector in4;

	if(idx*ReduceNum+ReduceNum-1>=valid_num && valid_num>0){
		in4= CondVectorGet<ReduceNum,DataType>::Get((DataType*)in_ptr,idx,valid_num);
	}else{
		in4=in_ptr[idx];
	}
	
	DataType current_max_value=max(in4);
	if(Offset!=0)
		out[Offset+idx]=current_max_value;

	ReduceLastAll<DataType,ReduceNumsAndOffset...> reduce_all(out,0,blockDim.x*gridDim.x);
	reduce_all.Run(current_max_value,idx,1,2);
}
};


template<int K>
void reduce_all_query(int& output_size,
						std::vector<int>& reduce_offset,
						int& topk_size,
						std::vector<int>& topk_size_offset,
						std::vector<int>& reduce_nums,
						int DataSize,int ReduceNum,bool reduce16=false){

	int current_reduce_num=ReduceNum;

	std::vector<int> Offset;

	output_size=0;	
	topk_size=0;

	int current_data_size=DataSize;
	int reversed_size_for_topk=0;

	bool last_flag=false;

	constexpr int TILE_SIZE=2;
	while(current_reduce_num>1 && TILE_SIZE*K<current_data_size && current_data_size>128*TILE_SIZE){

		bool run_flag=!(current_data_size/current_reduce_num <TILE_SIZE*128 || current_data_size/current_reduce_num<TILE_SIZE*K);


		if(!run_flag){
			current_reduce_num=current_reduce_num/2;
			last_flag=true;
		}else{

			reduce_nums.push_back(current_reduce_num);

			reduce_offset.push_back(output_size);
			topk_size_offset.push_back(reversed_size_for_topk);


			current_data_size=current_data_size/current_reduce_num;

			output_size+=current_data_size;

			if(current_reduce_num==2){
				current_reduce_num=4;
			}
			if(current_reduce_num>=8){
				current_reduce_num=2;
			}
			

			if(last_flag){
				reversed_size_for_topk+=K;
				topk_size+=K;
			}else{
				reversed_size_for_topk+=K;
				topk_size+=K;
			}

			
		}
	}

	if((ReduceNum==8 or reduce16) && reduce_offset.size()>2){
		
		auto sub_offset=reduce_offset[1];
		for(int i=2;i<reduce_offset.size();i++){
			reduce_offset[i]-=sub_offset;
		}
		reduce_offset[0]=0;
		reduce_offset[1]=0;
		output_size-=sub_offset;
	}
}

template<typename type>
void cal_reduce_all_data_sizes(std::vector<int>& DataSizes,
		std::vector<int>& threadIdx_counts, int& finnal_vaid_num,
		std::vector<int> ReduceNums,int valid_data_size=0,
		type input_buffer_num=1){



	int current_data_size=valid_data_size;

	for(int i=0;i<(ReduceNums.size()+2)/3;i++){
		int current_idx=i*3;

		int left=3;
		if(current_idx+3>=ReduceNums.size()){
			left=ReduceNums.size()-current_idx;
		}


		int next_data_size=current_data_size;

		DataSizes.push_back(next_data_size);

		threadIdx_counts.push_back((next_data_size+ReduceNums[current_idx]-1)/ReduceNums[current_idx]);

		for(int j=current_idx;j<current_idx+left;j++){
			

			next_data_size=(next_data_size+ReduceNums[j]-1)/ReduceNums[j];

		}
		// auto tmp_current_data_size=(current_data_size+ReduceNums[current_idx]-1)/ReduceNums[current_idx];

		

		current_data_size=next_data_size;
	
		
	}

	finnal_vaid_num=current_data_size;
}
template<int K,typename DataType=float,typename OutDataType=float>
int reduce_all(DataType** ins,int valid_data_size,int input_buffer_num,
		OutDataType* out,std::vector<CUfunction> cu_funcs,
		std::vector<int> ReduceNums,cudaStream_t& stream){

	std::vector<int> DataSizes;
	std::vector<int> threadIdx_counts;

	int finnal_valid_num;
	cal_reduce_all_data_sizes(DataSizes,threadIdx_counts,finnal_valid_num,ReduceNums,valid_data_size,input_buffer_num);

	OutDataType* current_output_buffer=out;
	
	int cu_fun_offset=0;
	for(int i=0;i<DataSizes.size();i++){
		int BLOCK_SIZE=256> threadIdx_counts[i]? threadIdx_counts[i]:256;

		int in_valid_data_size = DataSizes[i];
		int in_valid_tile_num = (DataSizes[i]*ReduceNums[i*3]-1)/ReduceNums[i*3]/input_buffer_num;

		if(i==0){
			if(input_buffer_num==1){
				
				if(cu_funcs.size()>DataSizes.size()){
					void *args[] = {&ins[0], &in_valid_data_size, &in_valid_tile_num, &current_output_buffer};		
					auto status=cuLaunchKernel(cu_funcs[0], (threadIdx_counts[0])/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
						stream, args, nullptr);

					if(threadIdx_counts[0]%BLOCK_SIZE!=0){
						int offset=((threadIdx_counts[0])/BLOCK_SIZE)*BLOCK_SIZE;

						void *args2[] = {&ins[0], &in_valid_data_size, &in_valid_tile_num,&offset,&current_output_buffer};		
						status=cuLaunchKernel(cu_funcs[1], 1, 1, 1, BLOCK_SIZE, 1, 1, 0, 
							stream, args2, nullptr);
					}
					cu_fun_offset=1;
				}else{
					void *args[] = {&ins[0], &in_valid_data_size, &in_valid_tile_num, &current_output_buffer};		
					auto status=cuLaunchKernel(cu_funcs[0], (threadIdx_counts[0]+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
						stream, args, nullptr);
				}
			}
			// if(input_buffer_num==2){
			// 	void *args[] = {&ins[0],&ins[1], &in_valid_data_size, &in_valid_tile_num,&current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+1)/2+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==3){
			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&in_valid_data_size, &in_valid_tile_num, &current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+2)/3+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==4){
			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&ins[3], &in_valid_data_size, &in_valid_tile_num,&current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+3)/4+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==5){

			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&ins[3],&ins[4], &in_valid_data_size, &in_valid_tile_num, &current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+4)/5+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==6){
			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&ins[3],&ins[4],&ins[5], &in_valid_data_size, &in_valid_tile_num,&current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+5)/6+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==7){
			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&ins[3],&ins[4],&ins[5],&ins[6], &in_valid_data_size, &in_valid_tile_num, &current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+6)/7+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==8){
			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&ins[3],&ins[4],&ins[5],&ins[6], &ins[7], &in_valid_data_size, &in_valid_tile_num,&current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+7)/8+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
			// if(input_buffer_num==9){
			// 	void *args[] = {&ins[0],&ins[1],&ins[2],&ins[3],&ins[4],&ins[5],&ins[6], &ins[7],&ins[8], &in_valid_data_size, &in_valid_tile_num,&current_output_buffer};		
			// 	cuLaunchKernel(cu_funcs[0], ((threadIdx_counts[0]+8)/9+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
			// 		stream, args, nullptr);
			// }
		}else{

			void *args1[] = {&current_output_buffer, &in_valid_data_size};
			auto status=cuLaunchKernel(cu_funcs[i+cu_fun_offset],
			 (threadIdx_counts[i]+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 
				stream, args1, nullptr);
		}


	}

	return finnal_valid_num;
}


#endif

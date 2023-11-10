#ifndef UTILS_NEW_H_

#ifdef __DLGPUC64__
#include <cu_ext.h>
#endif
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


#include <map>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>


#include "reduce_last.h"
#include "topk.h"

CUmodule module{nullptr};

void load_all_cases_fatbin(std::string target_file_name){
	CUresult err = cuModuleLoad(&module, target_file_name.c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "cuModule load data failed "  << std::endl;
	}
}


template<int K>
auto cal_get_templte_topk_select_all(int in_size,bool keep_first_write){
	std::stringstream func_name;
	func_name<<"topk_select_all_"<<keep_first_write<<"_"<<K<<"_"<<in_size;

	CUfunction func_get{};
			
	CUresult err = cuModuleGetFunction(&func_get, module, func_name.str().c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "load select_all cu_funcs failed "  << std::endl;
	}
	return func_get;
}
template<int K,bool IsSorted=true>
auto cal_get_templte_topk_select_input(int input_buffer_num,int input_buffer_block_size,
	int reduce_size=4){

	std::stringstream func_name;
	int align_data_size = 1<<(int)ceil(std::log2(input_buffer_block_size));
	func_name<<"topk_select_input_"<<reduce_size<<"_"<<K<<"_"<<input_buffer_num<<"_"<<align_data_size;
	
	//std::cout<<func_name.str()<<std::endl;
	CUfunction func_get{};
				
	CUresult err = cuModuleGetFunction(&func_get, module, func_name.str().c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "load select_input cu_funcs failed :"<<func_name.str()  << std::endl;
	}
	return func_get;
}

template<int K,bool IsSorted=true>
auto cal_get_templte_topk_select_input_half(int input_buffer_num,int input_buffer_block_size,
			int reduce_size=4){

	std::stringstream func_name;
	int align_data_size = 1<<(int)ceil(std::log2(input_buffer_block_size));
	func_name<<"topk_select_input_half_"<<reduce_size<<"_"<<K<<"_"<<input_buffer_num<<"_"<<align_data_size;

	CUfunction func_get{};
				
	CUresult err = cuModuleGetFunction(&func_get, module, func_name.str().c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "load select_input cu_funcs failed "  << std::endl;
	}
	return func_get;
}

auto cal_get_template_topk_reduce_all(std::vector<int> reduce_offsets,std::vector<int> reduce_nums,bool keep_first_write){

	std::stringstream func_name;
	func_name<<"topk_reduce_all_"<<keep_first_write;
	for(int j=0;j<reduce_nums.size();j++){
		func_name<<"_"<<reduce_nums[j]<<"_"<<reduce_offsets[j];
	}


	CUfunction func_get{};
	// std::cout<<"func_name:"<<func_name.str()<<std::endl;		
	CUresult err=cuModuleGetFunction(&func_get, module, func_name.str().c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "load reduce_all cu_funcs failed "  << std::endl;
	}
	return func_get;
}

auto cal_get_template_topk_reduce_all_half(std::vector<int> reduce_offsets,std::vector<int> reduce_nums,bool keep_first_write){

	std::stringstream func_name;
	func_name<<"topk_reduce_all_half_"<<keep_first_write;
	for(int j=0;j<reduce_nums.size();j++){
		func_name<<"_"<<reduce_nums[j]<<"_"<<reduce_offsets[j];
	}


	CUfunction func_get{};
	// std::cout<<"func_name:"<<func_name.str()<<std::endl;		
	CUresult err=cuModuleGetFunction(&func_get, module, func_name.str().c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "load reduce_all_half cu_funcs failed: "<<func_name.str() << std::endl;
	}
	return func_get;
}
auto cal_get_template_topk_reduce_all_half_bound_check(std::vector<int> reduce_offsets,std::vector<int> reduce_nums,bool keep_first_write){

	std::stringstream func_name;
	func_name<<"topk_reduce_all_half_neg";
	for(int j=0;j<reduce_nums.size();j++){
		func_name<<"_"<<reduce_nums[j]<<"_"<<reduce_offsets[j];
	}


	CUfunction func_get{};
	// std::cout<<"func_name:"<<func_name.str()<<std::endl;		
	CUresult err=cuModuleGetFunction(&func_get, module, func_name.str().c_str());
	if (err != CUDA_SUCCESS) {
    	std::cout << "load reduce_all_half cu_funcs failed: "<<func_name.str() << std::endl;
	}
	return func_get;
}

void compile_reduce_all(std::vector<CUfunction>& cu_funcs,std::vector<int>& DataSizes,
		std::vector<int> ReduceNums,std::vector<int> Offsets,int DataSize,int valid_data_size=0,
		int input_buffer_num=1,bool is_half=false,int first_reduce_num=4){

	bool keep_first_write = (first_reduce_num<=4);


	int current_data_size=valid_data_size;

	for(int i=0;i<(Offsets.size()+2)/3;i++){
		int current_idx=i*3;

		int left=3;
		if(current_idx+3>=Offsets.size()){
			left=Offsets.size()-current_idx;
		}

		std::vector<int> sub_offset;
		std::vector<int> sub_reduceNums;
		
		// int reduce_num=1;
		int next_data_size=current_data_size;

		for(int j=current_idx;j<current_idx+left;j++){
			// reduce_num*=ReduceNums[j];

			next_data_size=(next_data_size+ReduceNums[j]-1)/ReduceNums[j];

			sub_reduceNums.push_back(ReduceNums[j]);
			sub_offset.push_back(Offsets[j]);
		}
		auto tmp_current_data_size=(current_data_size+ReduceNums[current_idx]-1)/ReduceNums[current_idx];

		DataSizes.push_back(tmp_current_data_size);
		if(i==0 && is_half){

			auto func=cal_get_template_topk_reduce_all_half(sub_offset,sub_reduceNums,keep_first_write);
			cu_funcs.push_back(func);	

			if(keep_first_write==false){
				auto func_bound_checked=cal_get_template_topk_reduce_all_half_bound_check(sub_offset,
					sub_reduceNums,keep_first_write);

				cu_funcs.push_back(func_bound_checked);
			}

		}else{
			auto func=cal_get_template_topk_reduce_all(sub_offset,sub_reduceNums,keep_first_write or i!=0);
			cu_funcs.push_back(func);			
		}

		current_data_size=next_data_size;
	
		
	}
}


template<typename DataType>
struct GetWorkSpaceSizeDataType
{
	
};
template<>
struct GetWorkSpaceSizeDataType<float>
{
	constexpr static int size=sizeof(float);
};
template<>
struct GetWorkSpaceSizeDataType<float16>
{
	constexpr static int size=sizeof(float);
};
template<typename DataType,int K,typename DataIndexType=int,bool IsSorted=true>
class TopkHelper{//warning only support float,int
public:
	TopkHelper(int DataSize,int input_buffer_num=1,bool with_input_index=false){
		auto align_data_size=1<<(int)ceil(std::log2(DataSize));

		this->DataSize=DataSize;
		this->DataAlignSize=align_data_size;
		this->WithInputIndex=with_input_index;
		this->InputBufferNum=input_buffer_num;
		this->is_half=sizeof(DataType)==2;

		//<=24 because hw fail
		this->first_reduce_num=this->DataAlignSize<=2048*8 || K*16*2*4>24*1024 ? 4 : 16;

		
		assert(DataSize%input_buffer_num==0 && "don't support tile ddr....!");


		if(this->is_half && this->first_reduce_num==16){
			reduce_all_query<K>(ReduceSize,
						ReduceOffset,
						TopkSize,
						TopkOffset,
						ReduceNums,
						align_data_size,8,true);
		}else if(this->first_reduce_num==16){
			reduce_all_query<K>(ReduceSize,
						ReduceOffset,
						TopkSize,
						TopkOffset,
						ReduceNums,
						align_data_size,4,true);
		}else{
			reduce_all_query<K>(ReduceSize,
						ReduceOffset,
						TopkSize,
						TopkOffset,
						ReduceNums,
						align_data_size,4,false);
		}
		
	}
	TopkHelper(){
	}

	int CalWorkspace(){
		int DataSize=GetWorkSpaceSizeDataType<DataType>::size;

		return this->ReduceSize*DataSize+K*DataSize*2+K*sizeof(DataIndexType)*2;
	}
	int CalReduceWorkspace(){
		int DataSize=GetWorkSpaceSizeDataType<DataType>::size;
		return this->ReduceSize*DataSize;
	}


	void CompileTopkSelectAll(){

		FuncTopkSelectAll=cal_get_templte_topk_select_all<K>(this->DataAlignSize,this->first_reduce_num==4);
		if(this->is_half){
			FuncTopkSelectInput=cal_get_templte_topk_select_input_half<K,IsSorted>(this->InputBufferNum,
			this->DataSize/this->InputBufferNum,this->first_reduce_num);
		}else{
			FuncTopkSelectInput=cal_get_templte_topk_select_input<K,IsSorted>(this->InputBufferNum,
				this->DataSize/this->InputBufferNum,this->first_reduce_num);
		}
	}

	void CompileReduce(){

		compile_reduce_all(ReduceCuFuncs,ReduceDataSize,
			ReduceNums,ReduceOffset,this->DataAlignSize,this->DataSize,
			this->InputBufferNum,this->is_half,this->first_reduce_num);
	}

	int ReduceAllExecute(float* d_output_ptr,DataType** d_input_ptr,int valid_data_size,cudaStream_t& stream){
		return reduce_all<K,DataType,float>(d_input_ptr,valid_data_size,
				this->InputBufferNum,d_output_ptr,this->ReduceCuFuncs,
			this->ReduceNums,stream);
	}

	void TopkSelectAllExecute(DataIndexType* output_value_idx_ptr,
									float* d_input_ptr,DataIndexType* d_select_idx_ptr,
								cudaStream_t& stream){

		CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(d_input_ptr);
		CUdeviceptr input1 = reinterpret_cast<CUdeviceptr>(d_select_idx_ptr);
		CUdeviceptr input2= reinterpret_cast<CUdeviceptr>(output_value_idx_ptr);
	    void *args[] = {&input0, &input1, &input2};

	    constexpr int BLOCK_SIZE= K==1024 ? 512 : K;
	  	cuLaunchKernel(FuncTopkSelectAll, 1, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args, nullptr);
	}

	void TopkExecute(float* output_value_ptr,  DataIndexType* output_value_idx_ptr,
						int finnal_valid_num,float* d_value_ptr,
						cudaStream_t& stream){

		int current_data_size=ReduceSize-ReduceOffset[ReduceOffset.size()-1];
		float* d_current_value_ptr=d_value_ptr+ReduceOffset[ReduceOffset.size()-1];

		topk_lunch<K,false,float,DataIndexType>(d_current_value_ptr,
				output_value_ptr,output_value_idx_ptr,current_data_size,finnal_valid_num,stream);
			
	}

	void TopkSelectExecute(DataType* d_out_ptr,DataIndexType* d_out_idx_ptr,
							DataType** d_input_ptr,DataIndexType* d_select_idx_ptr,
							cudaStream_t& stream, int updataDatesize){

		
		CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(d_out_ptr);
		CUdeviceptr input1= reinterpret_cast<CUdeviceptr>(d_out_idx_ptr);
		CUdeviceptr input2= reinterpret_cast<CUdeviceptr>(d_select_idx_ptr);

		CUdeviceptr input3 = reinterpret_cast<CUdeviceptr>(d_input_ptr[0]);
		int size = updataDatesize;
		//std::cout<<"size: "<<size<<std::endl;
		//std::cout<<"inputBufferNum: "<<this->InputBufferNum<<std::endl;
		int total_threadIdx=this->first_reduce_num/4*K;

		total_threadIdx = total_threadIdx > 512 ? 512 :total_threadIdx;
		//std::cout<<"total_threadIdx: "<<total_threadIdx<<std::endl;

		if(this->InputBufferNum==1){
		    void *args[] = {&size, &input0, &input1, &input2, &input3};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==2){

			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==3){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5};
		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==4){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5,&input6};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==5){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5,&input6,&input7};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==6){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5,&input6,&input7,&input8};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==7){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);
			CUdeviceptr input9 = reinterpret_cast<CUdeviceptr>(d_input_ptr[6]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5,&input6,&input7,&input8,&input9};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==8){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);
			CUdeviceptr input9 = reinterpret_cast<CUdeviceptr>(d_input_ptr[6]);
			CUdeviceptr input10 = reinterpret_cast<CUdeviceptr>(d_input_ptr[7]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5,&input6,&input7,&input8,&input9,&input10};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==9){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);
			CUdeviceptr input9 = reinterpret_cast<CUdeviceptr>(d_input_ptr[6]);
			CUdeviceptr input10 = reinterpret_cast<CUdeviceptr>(d_input_ptr[7]);
			CUdeviceptr input11 = reinterpret_cast<CUdeviceptr>(d_input_ptr[8]);

		    void *args[] = {&size, &input0, &input1, &input2,&input3,&input4,&input5,&input6,&input7,&input8,&input9,&input10,
		    				&input11};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	}

	void TopkSelectExecute(DataType* d_out_ptr,DataIndexType* d_out_idx_ptr,
							DataType** d_input_ptr,DataIndexType* d_input_idx_ptr,
							DataIndexType* d_select_idx_ptr,
							cudaStream_t& stream, int updataDatesize){

		CUdeviceptr input0 = reinterpret_cast<CUdeviceptr>(d_out_ptr);
		CUdeviceptr input1= reinterpret_cast<CUdeviceptr>(d_out_idx_ptr);
		CUdeviceptr input2= reinterpret_cast<CUdeviceptr>(d_select_idx_ptr);

		CUdeviceptr input3 = reinterpret_cast<CUdeviceptr>(d_input_ptr[0]);

		CUdeviceptr input_raw_idx=reinterpret_cast<CUdeviceptr>(d_input_idx_ptr);
		int size = updataDatesize;
		//std::cout<<"size: "<<size<<std::endl;
		//std::cout<<"input_buffer_num: "<<InputBufferNum<<std::endl;
		int total_threadIdx=this->first_reduce_num/4*K;
		
		total_threadIdx= total_threadIdx==1024? 512: total_threadIdx;

		if(this->InputBufferNum==1){
		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==2){

			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==3){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);


		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,&input5};
		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==4){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);

		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,&input5,&input6};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==5){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);

		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,&input5,&input6,&input7};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==6){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);

		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,&input5,&input6,&input7,&input8};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==7){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);
			CUdeviceptr input9 = reinterpret_cast<CUdeviceptr>(d_input_ptr[6]);

		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,&input5,&input6,&input7,&input8,&input9};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	  	if(this->InputBufferNum==8){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);
			CUdeviceptr input9 = reinterpret_cast<CUdeviceptr>(d_input_ptr[6]);
			CUdeviceptr input10 = reinterpret_cast<CUdeviceptr>(d_input_ptr[7]);

		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,&input5,&input6,&input7,&input8,&input9,&input10};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}

	  	if(this->InputBufferNum==9){
			CUdeviceptr input4 = reinterpret_cast<CUdeviceptr>(d_input_ptr[1]);
			CUdeviceptr input5 = reinterpret_cast<CUdeviceptr>(d_input_ptr[2]);
			CUdeviceptr input6 = reinterpret_cast<CUdeviceptr>(d_input_ptr[3]);
			CUdeviceptr input7 = reinterpret_cast<CUdeviceptr>(d_input_ptr[4]);
			CUdeviceptr input8 = reinterpret_cast<CUdeviceptr>(d_input_ptr[5]);
			CUdeviceptr input9 = reinterpret_cast<CUdeviceptr>(d_input_ptr[6]);
			CUdeviceptr input10 = reinterpret_cast<CUdeviceptr>(d_input_ptr[7]);
			CUdeviceptr input11 = reinterpret_cast<CUdeviceptr>(d_input_ptr[8]);

		    void *args[] = {&size, &input0, &input1, &input2,&input_raw_idx,&input3,&input4,
		    	&input5,&input6,&input7,&input8,&input9,&input10,&input11};

		  	cuLaunchKernel(FuncTopkSelectInput, 1, 1, 1, total_threadIdx, 1, 1, 0, stream, args, nullptr);
	  	}
	}


	void Compile(){

		CompileTopkSelectAll();
		CompileReduce();
	}
	void Execute(DataType* finnal_output,DataIndexType* finnal_idx,
					DataType** d_input_ptr,
					void* workspace,cudaStream_t& stream, int updataDatesize){

		float* d_output_ptr=(float*)workspace;


		int finnal_valid_num=ReduceAllExecute(d_output_ptr,(DataType**)d_input_ptr,updataDatesize,stream);

		float* output_value_ptr=(float*)(((char*)workspace)+CalReduceWorkspace());
		int* output_value_idx_ptr=(int*)(output_value_ptr+2*K);

		TopkExecute(output_value_ptr,output_value_idx_ptr,finnal_valid_num,d_output_ptr,stream);
	
		int* finnal_output_value_idx_ptr= output_value_idx_ptr+K;

		float* finnal_input_ptr=d_output_ptr;
#if DEBUG
		cudaStreamSynchronize(stream);
		PrintfCudaMemory(output_value_ptr,K,16);
		PrintfCudaMemory(output_value_idx_ptr,K,16);
#endif

		if(ReduceOffset.size()>=2){
			TopkSelectAllExecute(finnal_output_value_idx_ptr,finnal_input_ptr,output_value_idx_ptr,stream);
#if DEBUG
			cudaStreamSynchronize(stream);
			PrintfCudaMemory(finnal_output_value_idx_ptr,K,16);

#endif

			TopkSelectExecute((DataType*)finnal_output,(DataIndexType*)finnal_idx,
				d_input_ptr,finnal_output_value_idx_ptr,stream, updataDatesize);
		}else{
			TopkSelectExecute((DataType*)finnal_output,(DataIndexType*)finnal_idx,
				d_input_ptr,output_value_idx_ptr,stream, updataDatesize);
		}

	}
	void Execute(DataType* finnal_output,DataIndexType* finnal_idx,
					DataType** d_input_ptr,DataIndexType* d_input_idx_ptr,
					void* workspace,cudaStream_t& stream, int updataDatesize){

		float* d_output_ptr=(float*)workspace;
		int finnal_valid_num=ReduceAllExecute(d_output_ptr,updataDatesize,(DataType**)d_input_ptr,stream);

		float* output_value_ptr=(float*)(((char*)workspace)+CalReduceWorkspace());
		int* output_value_idx_ptr=(int*)(output_value_ptr+2*K);

		TopkExecute(output_value_ptr,output_value_idx_ptr,finnal_valid_num,d_output_ptr,stream);
		int* finnal_output_value_idx_ptr= output_value_idx_ptr+K;

		// cudaStreamSynchronize(stream);
		// PrintfCudaMemory(output_value_ptr,K,16);

		float* finnal_input_ptr=d_output_ptr;
		if(ReduceOffset.size()>2){
			TopkSelectAllExecute(finnal_output_value_idx_ptr,finnal_input_ptr,output_value_idx_ptr,stream);

			TopkSelectExecute((DataType*)finnal_output,(DataIndexType*)finnal_idx,
				d_input_ptr,(DataIndexType*)d_input_idx_ptr,finnal_output_value_idx_ptr,stream);
		}else{
			TopkSelectExecute((DataType*)finnal_output,(DataIndexType*)finnal_idx,
				d_input_ptr,(DataIndexType*)d_input_idx_ptr,output_value_idx_ptr,stream);
		}
	}

private:
	int DataSize;
	int DataAlignSize;
	int InputBufferNum;
	bool WithInputIndex;
	bool is_half;
	int first_reduce_num;
	int ReduceSize{0};
	int TopkSize{0};
	std::vector<int> ReduceOffset;
	std::vector<int> TopkOffset;
	std::vector<int> ReduceNums;

	std::vector<CUfunction> ReduceCuFuncs;
	std::vector<int> ReduceDataSize;

	CUfunction FuncTopkSelectAll;

	CUfunction FuncTopkSelectInput;

};

static map<std::string,void*> topk_helpers;

template<typename DataType,typename DataIndexType,int K,bool IsSorted=true>
int topk_query_workspace(int DataSize,int input_buffer_num=1){
	stringstream key_stream;
	auto align_data_size=1<<(int)ceil(std::log2(DataSize));
	key_stream<<sizeof(DataType)<<"_"<<K<<"_"<<align_data_size<<"_"<<IsSorted<<"_"<<input_buffer_num;

	string key=key_stream.str();

	// std::cout<<"queryWorkSpaceSize:key("<<key<<")"<<std::endl;
	if(topk_helpers.find(key)==topk_helpers.end()){
		auto topk_helper= new TopkHelper<DataType,K,DataIndexType,IsSorted>(DataSize,input_buffer_num);
		topk_helper->Compile();
		topk_helpers[key]=topk_helper;	
	}
	auto topk_impl=(TopkHelper<DataType,K,DataIndexType,IsSorted>*)(topk_helpers.find(key)->second);

	return topk_impl->CalWorkspace();
}



template<typename DataType,typename DataIndexType,int K,bool IsSorted=true>
void topk_fun(DataType** data_buffer,int DataSize,
			DataType* output_buffer,DataIndexType* ouput_idx_buffer,
			void* workspace,cudaStream_t& stream,int input_buffer_num=1){

	stringstream key_stream;
	auto align_data_size=1<<(int)ceil(std::log2(DataSize));
	key_stream<<sizeof(DataType)<<"_"<<K<<"_"<<align_data_size<<"_"<<IsSorted<<"_"<<input_buffer_num;

	string key=key_stream.str();
	// std::cout<<"topk:key("<<key<<")"<<std::endl;

	if(topk_helpers.find(key)==topk_helpers.end()){
		auto topk_helper= new TopkHelper<DataType,K,DataIndexType,IsSorted>(DataSize,input_buffer_num);
		topk_helper->Compile();
		topk_helpers[key]=topk_helper;	
	}

	auto topk_impl=(TopkHelper<DataType,K,DataIndexType,IsSorted>*)(topk_helpers.find(key)->second);

	int updataDatesize = DataSize/input_buffer_num;
	topk_impl->Execute(output_buffer,ouput_idx_buffer,data_buffer,workspace,stream,updataDatesize);
}

template<typename DataType,typename DataIndexType,int K,bool IsSorted=true>
void topk_fun(DataType** data_buffer,DataIndexType* data_idx_buffer,
			int DataSize,
			DataType* output_buffer,DataIndexType* ouput_idx_buffer,
			void* workspace,cudaStream_t& stream,int input_buffer_num=1){

	stringstream key_stream;
	auto align_data_size=1<<(int)ceil(std::log2(DataSize));
	key_stream<<sizeof(DataType)<<"_"<<K<<"_"<<align_data_size<<"_"<<IsSorted<<"_"<<input_buffer_num;

	string key=key_stream.str();

	if(topk_helpers.find(key)==topk_helpers.end()){
		auto topk_helper= new TopkHelper<DataType,K,DataIndexType,IsSorted>(DataSize,input_buffer_num);
		topk_helper->Compile();
		topk_helpers[key]=topk_helper;	
	}

	auto topk_impl=(TopkHelper<DataType,K,DataIndexType,IsSorted>*)(topk_helpers.find(key)->second);

	int updataDatesize = DataSize/input_buffer_num;
	topk_impl->Execute(output_buffer,ouput_idx_buffer,data_buffer,data_idx_buffer,workspace,stream, updataDatesize);
}



/**for test interface**/
template<typename DataType = float, typename DataTypeIndex = int>
int queryTopkWorkSpaceSize(int k, int count, int input_buffer_num=1)
{
   
    int workspace_size;
    if(k==16){
    	workspace_size=topk_query_workspace<DataType,DataTypeIndex,16,true>(count,input_buffer_num);
    }else if(k==32){
    	workspace_size=topk_query_workspace<DataType,DataTypeIndex,32,true>(count,input_buffer_num);
    }else if(k==64){
    	workspace_size=topk_query_workspace<DataType,DataTypeIndex,64,true>(count,input_buffer_num);
    }else if(k==128){
    	workspace_size=topk_query_workspace<DataType,DataTypeIndex,128,true>(count,input_buffer_num);
    }else if(k==256){
        workspace_size=topk_query_workspace<DataType,DataTypeIndex,256,true>(count,input_buffer_num);
    }else if(k==512){
    	workspace_size=topk_query_workspace<DataType,DataTypeIndex,512,true>(count,input_buffer_num);
    }else if(k==1024){
    	workspace_size=topk_query_workspace<DataType,DataTypeIndex,1024,true>(count,input_buffer_num);
    }else{
    	workspace_size=-1;
    }

    return workspace_size;
}


template<int K, typename DataType = float, typename DataTypeIndex = int>
void topkk(DataType* input, DataTypeIndex* input_index, DataType* output, DataTypeIndex* output_index,
          void* work_space, int tile_size,cudaStream_t& stream){

	topk_fun<DataType,DataTypeIndex,K,true>(&input,input_index,tile_size,
			output,output_index,work_space,stream,1);
}

template<int K, typename DataType = float, typename DataTypeIndex = int>
void topkk(DataType* input, DataType* output, DataTypeIndex* output_index,
          void* work_space, int tile_size,cudaStream_t& stream){

	topk_fun<DataType,DataTypeIndex,K,true>(&input,tile_size,
			output,output_index,work_space,stream,1);
}


int queryTopkWorkSpaceSize(int k, int count, int input_buffer_num, bool isHalf)
{
    if(isHalf) {
        return queryTopkWorkSpaceSize<half, int>(k, count, input_buffer_num);
    } else {
        return queryTopkWorkSpaceSize<float, int>(k, count, input_buffer_num);
    }
}

void releaseTopkTmpMemory()
{
    for(auto iter = topk_helpers.begin(); iter != topk_helpers.end(); iter++)
    {
        std::string::size_type pos_dtype = iter->first.find("_");
        std::string::size_type pos_kvalue = iter->first.find("_", pos_dtype + 1);
        int dtype = atoi(iter->first.substr(0, pos_dtype).c_str());
        int kvalue = atoi(iter->first.substr(pos_dtype + 1, pos_kvalue).c_str());
        if(sizeof(half) == dtype) {
            if(16 == kvalue) {
                delete (TopkHelper<half, 16, int, true>*)(iter->second);
            } else if(32 == kvalue) {
                delete (TopkHelper<half, 32, int, true>*)(iter->second);
            } else if(64 == kvalue) {
                delete (TopkHelper<half, 64, int, true>*)(iter->second);
            } else if(128 == kvalue) {
                delete (TopkHelper<half, 128, int, true>*)(iter->second);
            } else if(256 == kvalue) {
                delete (TopkHelper<half, 256, int, true>*)(iter->second);
            } else if(512 == kvalue) {
                delete (TopkHelper<half, 512, int, true>*)(iter->second);
            } else if(1024 == kvalue) {
                delete (TopkHelper<half, 1024, int, true>*)(iter->second);
            }
        } else {
            if(16 == kvalue) {
                delete (TopkHelper<float, 16, int, true>*)(iter->second);
            } else if(32 == kvalue) {
                delete (TopkHelper<float, 32, int, true>*)(iter->second);
            } else if(64 == kvalue) {
                delete (TopkHelper<float, 64, int, true>*)(iter->second);
            } else if(128 == kvalue) {
                delete (TopkHelper<float, 128, int, true>*)(iter->second);
            } else if(256 == kvalue) {
                delete (TopkHelper<float, 256, int, true>*)(iter->second);
            } else if(512 == kvalue) {
                delete (TopkHelper<float, 512, int, true>*)(iter->second);
            } else if(1024 == kvalue) {
                delete (TopkHelper<float, 1024, int, true>*)(iter->second);
            }
        }
    }

    topk_helpers.erase(topk_helpers.begin(), topk_helpers.end());
}


void topk16(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<16>(input, output, output_index, work_space, tile_size, stream);
}

void topk16(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<16, half>(input, output, output_index, work_space, tile_size, stream);
}
void topk32(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<32>(input, output, output_index, work_space, tile_size, stream);
}

void topk32(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<32, half>(input, output, output_index, work_space, tile_size, stream);
}
void topk64(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<64>(input, output, output_index, work_space, tile_size, stream);
}

void topk64(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<64, half>(input, output, output_index, work_space, tile_size, stream);
}
void topk128(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<128>(input, output, output_index, work_space, tile_size, stream);
}

void topk128(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<128, half>(input, output, output_index, work_space, tile_size, stream);
}
void topk256(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<256>(input, output, output_index, work_space, tile_size, stream);
}

void topk256(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<256, half>(input, output, output_index, work_space, tile_size, stream);

}
void topk512(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<512>(input, output, output_index, work_space, tile_size, stream);
}

void topk512(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<512, half>(input, output, output_index, work_space, tile_size, stream);
}

void topk1024(float* input, float* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<1024>(input, output, output_index, work_space, tile_size, stream);
}

void topk1024(half* input, half* output, int* output_index, void* work_space,
            int tile_size, cudaStream_t& stream)
{
    topkk<1024, half>(input, output, output_index, work_space, tile_size, stream);
}

// template<int K, typename DataType = float, typename DataTypeIndex = int>
// void topkk(DataType** input, DataTypeIndex* input_index, DataType* output, DataTypeIndex* output_index,
//           void* work_space, int tile_size,cudaStream_t& stream, int input_buffer_num=1){

// 	topk_fun<DataType,DataTypeIndex,K,true>(input,input_index,tile_size,
// 			output,output_index,work_space,stream,input_buffer_num);
// }

// template<int K, typename DataType = float, typename DataTypeIndex = int>
// void topkk(DataType** input, DataType* output, DataTypeIndex* output_index,
//           void* work_space, int tile_size,cudaStream_t& stream, int input_buffer_num=1){

// 	topk_fun<DataType,DataTypeIndex,K,true>(input,tile_size,
// 			output,output_index,work_space,stream,input_buffer_num);
// }



#endif

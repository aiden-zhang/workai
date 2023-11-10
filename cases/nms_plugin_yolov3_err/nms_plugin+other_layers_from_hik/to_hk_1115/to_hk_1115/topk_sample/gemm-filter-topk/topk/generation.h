#ifndef GENERATION_H_
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include<set>

#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <vector>

using namespace std;

#define MODE (S_IRWXU | S_IRWXG | S_IRWXO)


int mk_dir(char* dir){
	DIR* mydir = NULL;
	if((mydir=opendir(dir))==NULL){
		int ret = mkdir(dir, MODE);
		if(ret != 0){
			printf("mkdir %s failed! \n", dir);
		}
		return ret;
	}
	return 1;
}



std::ofstream ofile;
std::set<std::string> files;

auto generate_cu(std::string source_content,std::string file_name){
	char tmp_dir[] = "tmp";
	
	std::string source_file="./tmp/"+file_name+".cu";

	if(files.count(file_name)==0){
		ofile<<source_content;	
	}
	files.insert(file_name);


}


void generate_topk_select_all_fstream(int K, std::vector<int> reduce_offsets,
	std::vector<int> reduce_nums,int in_size,bool isHaveFrist=true){
	std::stringstream func_name;
	func_name<<"topk_select_all_"<<isHaveFrist<<"_"<<K<<"_"<<in_size;

	std::stringstream func_template;
    func_template<<"\n#include \"topk/topk.h\""<<std::endl;

	func_template<<"extern \"C\" __global__ void "<<func_name.str()
					<<"(float* __restrict__ reduce_data_buffer,int* __restrict__ selected_idx,int* __restrict__ outputIdx){\n"
					<<"\ttopk_all_select_device_fun<"<<K<<",float,int";

	for(int j=reduce_offsets.size()-2;j>=0;j--){
		if(isHaveFrist==false && j==0){
			break;
		}
		func_template<<","<<reduce_nums[j+1]<<","<<reduce_offsets[j];
	}
	func_template<<">(reduce_data_buffer,selected_idx,outputIdx);\n}";

				
	generate_cu(func_template.str(),func_name.str());
}

void generate_topk_select_input_fstream(int K, int reduce_nums,int input_buffer_num,
			int input_buffer_block_size,bool with_input_index=false, bool IsSorted=true){
	std::stringstream func_name;
	int align_data_size = 1<<(int)ceil(std::log2(input_buffer_block_size));
	func_name<<"topk_select_input_"<<reduce_nums<<"_"<<K<<"_"<<input_buffer_num<<"_"<<align_data_size;

	std::stringstream func_template;
    func_template<<"\n#include \"topk/topk.h\""<<std::endl;

	func_template<<"extern \"C\" __global__ void "<<func_name.str()
					<<"(int input_buffer_block_size,float* __restrict__ output_buffer,int* __restrict__ output_idx_buffer,int* __restrict__ selected_idx";

	if(with_input_index){
		func_template<<",int* __restrict__ input_index";
	}

	for(int i=0;i<input_buffer_num;i++){
		func_template<<",float* __restrict__ input_buffer_"<<i;
	}

	func_template<<"){\n"
	<<"\tTopkOperation<"<<IsSorted;
	if(reduce_nums==4){
		func_template<<",float4";
	}else if(reduce_nums==16){
		func_template<<",float32x16";
	}else{
		func_template<<",float2";
	}

	func_template<<",int>::template Run<"<<K<<", 1,"<<K;

	for(int i=0;i<input_buffer_num;i++){
		func_template<<",float* __restrict__";
	}

	func_template<< ">(input_buffer_block_size,output_buffer,output_idx_buffer,selected_idx";

	if(with_input_index){
		func_template<<",input_index";
	}

	for(int i=0;i<input_buffer_num;i++){
		func_template<< ",input_buffer_"<<i;
	}
	func_template<<");\n}";

	generate_cu(func_template.str(),func_name.str());

}

void generate_topk_select_input_half_fstream(int K, int reduce_nums,int input_buffer_num,
			int input_buffer_block_size,bool with_input_index=false, bool IsSorted=true){
	std::stringstream func_name;
	int align_data_size = 1<<(int)ceil(std::log2(input_buffer_block_size));
	func_name<<"topk_select_input_half_"<<reduce_nums<<"_"<<K<<"_"<<input_buffer_num<<"_"<<align_data_size;

	std::stringstream func_template;
    func_template<<"\n#include \"topk/topk.h\""<<std::endl;

	func_template<<"extern \"C\" __global__ void "<<func_name.str()
					<<"(int input_buffer_block_size,half* __restrict__ output_buffer,int* __restrict__ output_idx_buffer,int* __restrict__ selected_idx";

	if(with_input_index){
		func_template<<",int* __restrict__ input_index";
	}

	for(int i=0;i<input_buffer_num;i++){
		func_template<<",half* __restrict__ input_buffer_"<<i;
	}

	func_template<<"){\n"
	<<"\tTopkOperation<"<<IsSorted;
	if(reduce_nums==4){
		func_template<<",float16x4";
	}else if(reduce_nums==16){
		func_template<<",float16x16";

	}else if(reduce_nums==8){
		func_template<<",float16x8";

	}else{
		func_template<<",float16x2";
	}

	func_template<<",int>::template Run<"<<K<<", 1,"<<K;

	for(int i=0;i<input_buffer_num;i++){
		func_template<<",half* __restrict__";
	}

	func_template<< ">(input_buffer_block_size,output_buffer,output_idx_buffer,selected_idx";

	if(with_input_index){
		func_template<<",input_index";
	}

	for(int i=0;i<input_buffer_num;i++){
		func_template<< ",input_buffer_"<<i;
	}
	func_template<<");\n}";

	generate_cu(func_template.str(),func_name.str());

}
void generate_topk_reduce_item_fstream(std::vector<int> reduce_offsets,std::vector<int> reduce_nums,
			int input_buffer_nums=1,int first_offset=0,bool frist_write=true){
	std::stringstream func_name;
	func_name<<"topk_reduce_all_"<<frist_write;
	for(int j=0;j<reduce_nums.size();j++){
		func_name<<"_"<<reduce_nums[j]<<"_"<<reduce_offsets[j];
	}

	std::stringstream func_template;
    func_template<<"\n#include \"topk/reduce_last.h\""<<std::endl;

	func_template<<"extern \"C\" __global__ void "<<func_name.str();

	if(reduce_offsets[0]==0){
		if(input_buffer_nums==0){
			func_template<<"(float* __restrict__ input_buffer,int valid_data_size,int tile_num,float* __restrict__ out_buffer){\n";
		}else{
			func_template<<"(";
			for(int j=0;j<input_buffer_nums;j++){
				func_template<<"float* __restrict__ input_buffer_"<<j<<",";
			}
			func_template<<"int valid_data_size,int tile_num,";
			func_template<<"float* __restrict__ output_buffer){\n";
		}
	}else{
		func_template<<"(float* __restrict__ out_buffer, int valid_data_size){\n";
	}
	func_template<<"\tReduceLastImpl<float,"<<frist_write;

	for(int j=0;j<reduce_nums.size();j++){
		func_template<<","<<reduce_nums[j]<<","<<reduce_offsets[j];
	}

	if(reduce_offsets[0]==0){
		if(input_buffer_nums==0){
			

			func_template<<">::ReduceLastAllDevice(input_buffer,out_buffer";

			func_template<<",valid_data_size";
			
		}else{
			func_template<<">::template ReduceLastAllDevice<0";

			for(int j=0;j<input_buffer_nums;j++){
				func_template<<",float* __restrict__";
			}

			func_template<<">(output_buffer"<<",valid_data_size,tile_num";
			for(int j=0;j<input_buffer_nums;j++){
				func_template<<",input_buffer_"<<j;
			}
		}
		func_template<<");\n}\n";
	}else{
		func_template<<">::template ReduceLastAllDevice<"<<first_offset<<">(out_buffer,valid_data_size);\n}\n";
	}
				
generate_cu(func_template.str(),func_name.str());


}

void generate_topk_reduce_half_bound_check_item_fstream(std::vector<int> reduce_offsets,std::vector<int> reduce_nums,
			int input_buffer_nums=1,int first_offset=0,bool frist_write=true){
	std::stringstream func_name;
	func_name<<"topk_reduce_all_half_neg";

	for(int j=0;j<reduce_nums.size();j++){
		func_name<<"_"<<reduce_nums[j]<<"_"<<reduce_offsets[j];
	}

	std::stringstream func_template;
    func_template<<"\n#include \"topk/reduce_last.h\""<<std::endl;

	func_template<<"extern \"C\" __global__ void "<<func_name.str();

	if(reduce_offsets[0]==0){
		if(input_buffer_nums==0){
			func_template<<"(half* __restrict__ input_buffer,int valid_data_size,int tile_num,int offset,float* __restrict__ out_buffer){\n";
		}else{
			func_template<<"(";
			for(int j=0;j<input_buffer_nums;j++){
				func_template<<"half* __restrict__ input_buffer_"<<j<<",";
			}
			func_template<<"int valid_data_size,int tile_num,int offset,";
			func_template<<"float* __restrict__ output_buffer){\n";
		}
	}else{
		func_template<<"(float* __restrict__ out_buffer, int valid_data_size){\n";
	}
	func_template<<"\tReduceLastImpl<half,"<<-1;

	for(int j=0;j<reduce_nums.size();j++){
		func_template<<","<<reduce_nums[j]<<","<<reduce_offsets[j];
	}

	if(reduce_offsets[0]==0){
		
		func_template<<">::template ReduceLastAllDevice<0";

		for(int j=0;j<input_buffer_nums;j++){
			func_template<<",half* __restrict__";
		}

		func_template<<">(output_buffer"<<",valid_data_size,tile_num,offset";
		for(int j=0;j<input_buffer_nums;j++){
			func_template<<",input_buffer_"<<j;
		}
		
		func_template<<");\n}\n";
	}else{
		func_template<<">::template ReduceLastAllDevice<"<<first_offset<<">(out_buffer,valid_data_size);\n}\n";
	}

	generate_cu(func_template.str(),func_name.str());
}

void generate_topk_reduce_half_item_fstream(std::vector<int> reduce_offsets,std::vector<int> reduce_nums,
			int input_buffer_nums=1,int first_offset=0,bool frist_write=true){
	

	if(input_buffer_nums==1 && frist_write==false && first_offset==0 && reduce_nums[0]==8)
		generate_topk_reduce_half_bound_check_item_fstream(reduce_offsets,reduce_nums,
			input_buffer_nums,first_offset,frist_write);

	std::stringstream func_name;
	func_name<<"topk_reduce_all_half_"<<frist_write;

	for(int j=0;j<reduce_nums.size();j++){
		func_name<<"_"<<reduce_nums[j]<<"_"<<reduce_offsets[j];
	}

	std::stringstream func_template;
    func_template<<"\n#include \"topk/reduce_last.h\""<<std::endl;

	func_template<<"extern \"C\" __global__ void "<<func_name.str();

	if(reduce_offsets[0]==0){
		if(input_buffer_nums==0){
			func_template<<"(half* __restrict__ input_buffer,int valid_data_size,int tile_num,float* __restrict__ out_buffer){\n";
		}else{
			func_template<<"(";
			for(int j=0;j<input_buffer_nums;j++){
				func_template<<"half* __restrict__ input_buffer_"<<j<<",";
			}
			func_template<<"int valid_data_size,int tile_num,";
			func_template<<"float* __restrict__ output_buffer){\n";
		}
	}else{
		func_template<<"(float* __restrict__ out_buffer, int valid_data_size){\n";
	}
	func_template<<"\tReduceLastImpl<half,"<<frist_write;

	for(int j=0;j<reduce_nums.size();j++){
		func_template<<","<<reduce_nums[j]<<","<<reduce_offsets[j];
	}

	if(reduce_offsets[0]==0){
		if(input_buffer_nums==0){
			

			func_template<<">::ReduceLastAllDevice(input_buffer,out_buffer";

			func_template<<",valid_data_size";
			
		}else{
			func_template<<">::template ReduceLastAllDevice<0";

			for(int j=0;j<input_buffer_nums;j++){
				func_template<<",half* __restrict__";
			}

			func_template<<">(output_buffer"<<",valid_data_size,tile_num";
			for(int j=0;j<input_buffer_nums;j++){
				func_template<<",input_buffer_"<<j;
			}
		}
		func_template<<");\n}\n";
	}else{
		func_template<<">::template ReduceLastAllDevice<"<<first_offset<<">(out_buffer,valid_data_size);\n}\n";
	}

	generate_cu(func_template.str(),func_name.str());

}

void generate_topk_reduce_all_fstream(std::vector<int> ReduceNums,std::vector<int> Offsets,
		int DataSize,int valid_data_size=0,
		int input_buffer_num=1){


	int current_data_size=valid_data_size;//ReduceNums[0];

	for(int i=0;i<(Offsets.size()+2)/3;i++){
		int current_idx=i*3;

		int left=3;
		if(current_idx+3>=Offsets.size()){
			left=Offsets.size()-current_idx;
		}

		std::vector<int> sub_offset;
		std::vector<int> sub_reduceNums;
		
		int next_data_size=current_data_size;

		for(int j=current_idx;j<current_idx+left;j++){

			next_data_size=(next_data_size+ReduceNums[j]-1)/ReduceNums[j];

			sub_reduceNums.push_back(ReduceNums[j]);
			sub_offset.push_back(Offsets[j]);
		}
		auto tmp_current_data_size=(current_data_size+ReduceNums[current_idx]-1)/ReduceNums[current_idx];


		if(sub_reduceNums[0]<8)
			generate_topk_reduce_item_fstream(sub_offset,sub_reduceNums,//current_data_size,
					input_buffer_num,current_idx >0 ? Offsets[current_idx-1]:0,true);

		if(i==0){
			generate_topk_reduce_half_item_fstream(sub_offset,sub_reduceNums,//current_data_size,
						input_buffer_num,current_idx >0 ? Offsets[current_idx-1]:0,false);

			if(sub_reduceNums[0]<8){
				generate_topk_reduce_item_fstream(sub_offset,sub_reduceNums,//current_data_size,
					input_buffer_num,current_idx >0 ? Offsets[current_idx-1]:0,false);

				generate_topk_reduce_half_item_fstream(sub_offset,sub_reduceNums,//current_data_size,
						input_buffer_num,current_idx >0 ? Offsets[current_idx-1]:0,true);
			}
		}

		

		current_data_size=next_data_size;
		
	}
}


void query_reduce_size(int K, int& output_size,
						std::vector<int>& reduce_offset,
						int& topk_size,
						std::vector<int>& topk_size_offset,
						std::vector<int>& reduce_nums,
						int DataSize,int ReduceNum,bool IsReduce16=false){


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

			if(last_flag){
				reversed_size_for_topk+=K;
				topk_size+=K;
			}else{
				reversed_size_for_topk+=K;
				topk_size+=K;
			}

			if(current_reduce_num==2){
				current_reduce_num=4;
			}
			if(current_reduce_num>=8){
				current_reduce_num=2;
			}
			
		}
	}


	if((ReduceNum==8 or IsReduce16) && reduce_offset.size()>2){
		
		auto sub_offset=reduce_offset[1];
		for(int i=2;i<reduce_offset.size();i++){
			reduce_offset[i]-=sub_offset;
		}
		reduce_offset[0]=0;
		reduce_offset[1]=0;

		output_size=output_size-sub_offset;
	}


}

void GenerateCasefloat16x16(int K, int DataSize, int InputBufferNum=1){
	/***
		warning only for half8,mean reduce nums 8 2,but offset 0,0,....
	***/
	
	
	assert(DataSize%InputBufferNum==0 && "don't support tile ddr....!");
	int DataAlignSize = 1<<(int)ceil(std::log2(DataSize));
	//std::cout<<"DataAlignSize: "<<DataAlignSize<<std::endl;
	int ReduceSize = 0;
	int TopkSize = 0;
	bool WithInputIndex = false;
	bool IsSorted = true;
	std::vector<int> ReduceOffset;
	std::vector<int> TopkOffset;
	std::vector<int> ReduceNums;

	query_reduce_size(K, ReduceSize, ReduceOffset, TopkSize, TopkOffset, ReduceNums, DataAlignSize, 8,true);

	generate_topk_select_all_fstream(K, ReduceOffset, ReduceNums, DataAlignSize,false);


	generate_topk_select_input_half_fstream(K, 16, InputBufferNum, DataSize/InputBufferNum,
		 WithInputIndex, IsSorted);

	generate_topk_reduce_all_fstream(ReduceNums, ReduceOffset, DataAlignSize, DataSize, InputBufferNum);
	
}

void GenerateCasefloat32x16(int K, int DataSize, int InputBufferNum=1){
	/***
		warning only for float4,mean reduce nums 4 4,but offset 0,0,....
	***/
	
	
	assert(DataSize%InputBufferNum==0 && "don't support tile ddr....!");
	int DataAlignSize = 1<<(int)ceil(std::log2(DataSize));
	//std::cout<<"DataAlignSize: "<<DataAlignSize<<std::endl;
	int ReduceSize = 0;
	int TopkSize = 0;
	bool WithInputIndex = false;
	bool IsSorted = true;
	std::vector<int> ReduceOffset;
	std::vector<int> TopkOffset;
	std::vector<int> ReduceNums;

	query_reduce_size(K, ReduceSize, ReduceOffset, TopkSize, TopkOffset, ReduceNums, DataAlignSize, 4,true);


	generate_topk_select_all_fstream(K, ReduceOffset, ReduceNums, DataAlignSize,false);

	generate_topk_select_input_fstream(K, 16, InputBufferNum, 
					DataSize/InputBufferNum,
					WithInputIndex, IsSorted);
	

	generate_topk_reduce_all_fstream(ReduceNums, ReduceOffset, DataAlignSize, DataSize, InputBufferNum);
	
}

void GenerateCase(int K, int DataSize, int InputBufferNum=1){

	
	if(K*16*2*4<=1024*24){ //<=24 because hw fail
		GenerateCasefloat16x16(K,DataSize,InputBufferNum);
		GenerateCasefloat32x16(K,DataSize,InputBufferNum);
    }

	assert(DataSize%InputBufferNum==0 && "don't support tile ddr....!");
	int DataAlignSize = 1<<(int)ceil(std::log2(DataSize));
	//std::cout<<"DataAlignSize: "<<DataAlignSize<<std::endl;
	int ReduceSize = 0;
	int TopkSize = 0;
	bool WithInputIndex = false;
	bool IsSorted = true;
	std::vector<int> ReduceOffset;
	std::vector<int> TopkOffset;
	std::vector<int> ReduceNums;


	query_reduce_size(K, ReduceSize, ReduceOffset, TopkSize,
		 TopkOffset, ReduceNums, DataAlignSize, 4,false);

	generate_topk_select_all_fstream(K, ReduceOffset, ReduceNums, DataAlignSize,true);
	// generate_topk_select_all_fstream(K, ReduceOffset, ReduceNums, DataAlignSize,false);


	generate_topk_select_input_fstream(K, ReduceNums[0], InputBufferNum, 
					DataSize/InputBufferNum,
					WithInputIndex, IsSorted);
	

	
	generate_topk_select_input_half_fstream(K, ReduceNums[0], InputBufferNum,
		 DataSize/InputBufferNum,WithInputIndex, IsSorted);


	generate_topk_reduce_all_fstream(ReduceNums, ReduceOffset, DataAlignSize,
		 DataSize, InputBufferNum);
	
}


#endif

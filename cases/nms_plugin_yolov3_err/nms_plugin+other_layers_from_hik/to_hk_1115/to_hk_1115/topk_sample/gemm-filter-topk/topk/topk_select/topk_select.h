/*
 * topk_select.h
 *
 *  Created on: 2021年11月10日
 *      Author: xiao.luo
 */

#ifndef TOPK_SELECT_H_
#define TOPK_SELECT_H_
#include<cuda.h>
#include<stdlib.h>
#include<assert.h>
#include "bin_count.h"

//DataType is must uchar
template<typename DataType>
class TopkBinSelect {
public:
	static constexpr int count_block_size = 512;
	static constexpr int csum_block_size = 256;
	static constexpr int grid_num_128 = 128;
	static constexpr int grid_num_64 = 64;
	static constexpr int grid_num_32 = 32;
	static constexpr int grid_num_16 = 16;
	static constexpr int grid_num_8 = 8;
	static constexpr int grid_num_4 = 4;

	TopkBinSelect(int data_size) {
		assert(data_size % 16 == 0 && "fail....");
		assert(data_size / 16 >= count_block_size * 4 && "fail...");

		this->valid_num = data_size / 16;
		this->GetTileBlockGridDimX();
	}

	void GetTileBlockGridDimX() {
		int grid_size = (this->valid_num + count_block_size - 1)
				/ count_block_size;

		if (grid_size > grid_num_128) {
			this->tile_block = (this->valid_num
					+ count_block_size * grid_num_128 - 1)
					/ (count_block_size * grid_num_128);
			this->grid_size = grid_num_128;
		} else if (grid_size > grid_num_64) {
			this->tile_block = (this->valid_num + count_block_size * grid_num_64
					- 1) / (count_block_size * grid_num_64);
			this->grid_size = grid_num_64;
		} else if (grid_size > grid_num_32) {
			this->tile_block = (this->valid_num + count_block_size * grid_num_32
					- 1) / (count_block_size * grid_num_32);
			this->grid_size = grid_num_32;
		} else if (grid_size > grid_num_16) {
			this->tile_block = (this->valid_num + count_block_size * grid_num_16
					- 1) / (count_block_size * grid_num_16);
			this->grid_size = grid_num_16;
		} else if (grid_size > grid_num_8) {
			this->tile_block = (this->valid_num + count_block_size * grid_num_8
					- 1) / (count_block_size * grid_num_8);
			this->grid_size = grid_num_8;
		} else if (grid_size > grid_num_4) {
			this->tile_block = (this->valid_num + count_block_size * grid_num_4
					- 1) / (count_block_size * grid_num_4);
			this->grid_size = grid_num_4;
		}

	}
	int QueryWorkSapce() {
		return this->grid_size * csum_block_size * sizeof(int)
				+ (this->grid_size + 1) * sizeof(int);
	}

	void Execute(DataType* hmdis, int K, int* topkIdx, void* workspace,cudaStream_t& stream) {
		dim3 grid(this->grid_size, 1, 1);
		dim3 block(this->count_block_size, 1, 1);

		int* bin_counts = (int*) workspace;
		int* csum_bin_counts = bin_counts + this->grid_size * csum_block_size;

		this->BinCountLaunch(this->tile_block,grid,block,hmdis,bin_counts,stream);

		//cudaDeviceSynchronize();

		this->CountSumPrefixSumLaunch(bin_counts,K,csum_bin_counts,stream);

		//DebugBuffer(csum_bin_counts,this->grid_size+1);
		//cudaDeviceSynchronize();

		this->TopkMoveHeadLaunch(this->tile_block,grid,block,hmdis,csum_bin_counts,K,topkIdx,stream);
	}
private:
	void DebugBuffer(int* buffer,int size,int limit_n=64){
			int* host_buffer = (int*) malloc(sizeof(int) * size);
			cudaMemcpy(host_buffer, buffer, sizeof(int) * size, cudaMemcpyDeviceToHost);
			std::cout << "printf_cuda_buffer:[";

			for (int i = 0; i < size; i++) {
				if (i % 64 == 0) {
					std::cout << std::endl;
				}
				std::cout << host_buffer[i] << "\t";

				if (i > 64 * limit_n) {
					break;
				}
			}
			std::cout << "]" << std::endl;

			free(host_buffer);
	}
	void BinCountLaunch(int tile_block,dim3& grid,dim3 block,void* hmdist,int* bin_counts,cudaStream_t& stream) {
		switch (tile_block) {
		case 1 :bin_count_op<csum_block_size, count_block_size, 1 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,bin_counts);break;
		case 2 :bin_count_op<csum_block_size, count_block_size, 2 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,bin_counts);break;
		case 3 :bin_count_op<csum_block_size, count_block_size, 3 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,bin_counts);break;
		case 4 :bin_count_op<csum_block_size, count_block_size, 4 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,bin_counts);break;
		default :bin_count_op<csum_block_size, count_block_size, -1 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,bin_counts);break;
		

		}
	}
	void CountSumPrefixSumLaunch(int* bin_counts,int K,int* csum_bin_counts,cudaStream_t& stream) {
		switch(this->grid_size/4) {
			case 1:count_sum_prefixsum_op<csum_block_size, csum_block_size,1> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;
			case 2:count_sum_prefixsum_op<csum_block_size, csum_block_size,2> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;
			case 4:count_sum_prefixsum_op<csum_block_size, csum_block_size,4> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;
			case 8:count_sum_prefixsum_op<csum_block_size, csum_block_size,8> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;
			case 16:count_sum_prefixsum_op<csum_block_size, csum_block_size,16> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;
			case 32:count_sum_prefixsum_op<csum_block_size, csum_block_size,32> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;
			case 64:count_sum_prefixsum_op<csum_block_size, csum_block_size,64> <<<dim3(1,1,1),dim3(csum_block_size,1,1),0,stream>>>(bin_counts,K,csum_bin_counts);break;

		}

	}
	void TopkMoveHeadLaunch(int tile_block,dim3& grid,dim3 block,void* hmdist,int* csum_bin_counts,int K,int* topkIdx,cudaStream_t& stream) {
		switch(tile_block) {
		case 1 :topk_movehead_op<count_block_size, 1 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,csum_bin_counts,K,topkIdx);break;
		case 2 :topk_movehead_op<count_block_size, 2 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,csum_bin_counts,K,topkIdx);break;
		case 3 :topk_movehead_op<count_block_size, 3 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,csum_bin_counts,K,topkIdx);break;
		case 4 :topk_movehead_op<count_block_size, 4 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,csum_bin_counts,K,topkIdx);break;
		default :topk_movehead_op<count_block_size, -1 ><<<grid,block,0,stream>>>(hmdist,this->valid_num,csum_bin_counts,K,topkIdx);break;
		
		}
	}

	int valid_num;
	int tile_block;
	int grid_size;
};


template<int K=10000>
void topkk(unsigned char* input,void* work_space,int total_size,int* topkIdx,
          cudaStream_t& stream){

	TopkBinSelect<unsigned char> topk_select(total_size);
	topk_select.Execute(input,K,topkIdx,work_space,stream);
}


int queryTopkWorkSpaceSize(int total_size)
{
    TopkBinSelect<unsigned char> topk_select(total_size);
    return topk_select.QueryWorkSapce();
}


void topk10000(unsigned char* input,int* output_index, void* work_space,
            int total_size, cudaStream_t& stream){
	topkk<10000>(input,work_space,total_size,output_index,stream);
}
#endif /* TOPK_SELECT_H_ */

/*
 * bin_count.h
 *
 *  Created on: 2021年11月9日
 *      Author: xiao.luo
 */

#include<stdio.h>
#include<cuda.h>
#include "csum_block.h"
#ifndef BIN_COUNT_H_
#define BIN_COUNT_H_

template<int LoopCountIdx>
struct LoopCountShare {
	__device__ static void run(int* sm, unsigned char* value_chars) {
		int bin_id = value_chars[LoopCountIdx];
		atomicAdd(&sm[bin_id], 1);
		LoopCountShare<LoopCountIdx - 1>::run(sm, value_chars);
	}
	__device__ static void write(int* topkIdx, int* global_buffer_idx,
			unsigned char* value_chars, int tidx, int select_bin_id, int K) {
		int bin_id = value_chars[LoopCountIdx];
		if (bin_id <= select_bin_id) {
			int global_address = atomicAdd(&global_buffer_idx[0], 1);
			if (global_address < K)
				topkIdx[global_address] = tidx * 16 + LoopCountIdx;
		}
		LoopCountShare<LoopCountIdx - 1>::write(topkIdx, global_buffer_idx,
				value_chars, tidx, select_bin_id, K);
	}
};
template<>
struct LoopCountShare<-1> {
	__device__ static void run(int* sm, unsigned char* value_chars) {
	}
	__device__ static void write(int* topkIdx, int* global_buffer_idx,
			unsigned char* value, int tidx, int select_bin_id, int K) {

	}
};


__device__ void update_sm(int* sm, unsigned int value) {
	atomicAdd(&sm[value & 255], 1);
	atomicAdd(&sm[(value>>8) & 255], 1);
	atomicAdd(&sm[(value>>16) & 255], 1);
	atomicAdd(&sm[(value>>24)], 1);
}

__device__ void update_write_global_buffer(int* topkIdx, int* global_buffer_idx,
			 unsigned int value, int tidx, int select_bin_id, int K) {
	
	int bin_id = value>>24;
	if (bin_id <= select_bin_id) {
		int global_address = atomicAdd(&global_buffer_idx[0], 1);
		if (global_address < K)
				topkIdx[global_address] = tidx + 3;
	}
	bin_id = (value>>16)&255;
	if (bin_id <= select_bin_id) {
		int global_address = atomicAdd(&global_buffer_idx[0], 1);
		if (global_address < K)
				topkIdx[global_address] = tidx + 2;
	}
	bin_id = (value>>8)&255;
	if (bin_id <= select_bin_id) {
		int global_address = atomicAdd(&global_buffer_idx[0], 1);
		if (global_address < K)
				topkIdx[global_address] = tidx + 1;
	}
	bin_id = value&255;
	if (bin_id <= select_bin_id) {
		int global_address = atomicAdd(&global_buffer_idx[0], 1);
		if (global_address < K)
				topkIdx[global_address] = tidx + 0;
	}
}
//warning BIN_SIZE=256, BLOCK_SIZE is  1024,gridDim.x <=32
template<int BIN_SIZE, int BLOCK_SIZE, int BLOCK_DIM_TILE>
__global__ void bin_count_op( void* __restrict__ hmdist, int valid_num, int* __restrict__ bin_counts) {

	__shared__  int bin_count[BIN_SIZE];
	if (threadIdx.x < BIN_SIZE){
		bin_count[threadIdx.x] = 0;
    }
	__syncthreads();
	if(BLOCK_DIM_TILE>0){
		for (int tileIdx = 0; tileIdx < BLOCK_DIM_TILE; tileIdx++) {
			int tidx = threadIdx.x + BLOCK_SIZE * blockIdx.x * BLOCK_DIM_TILE
					+ BLOCK_SIZE * tileIdx;
			if (tidx < valid_num) {
				auto value = ((uint4*) hmdist)[tidx];
				update_sm(bin_count,value.x);
				update_sm(bin_count,value.y);
				update_sm(bin_count,value.z);
				update_sm(bin_count,value.w);
				//LoopCountShare<15>::run(bin_count, (unsigned char*) &value);
			}
		}
	}else{
		for (int tidx = threadIdx.x+blockIdx.x*BLOCK_SIZE; tidx < valid_num; tidx+=BLOCK_SIZE*gridDim.x){
		
			auto value = ((uint4*) hmdist)[tidx];
			update_sm(bin_count,value.x);
			update_sm(bin_count,value.y);
			update_sm(bin_count,value.z);
			update_sm(bin_count,value.w);
			//LoopCountShare<15>::run(bin_count, (unsigned char*) &value);
			
		}
	}
	__syncthreads();
	if (threadIdx.x < BIN_SIZE)
		bin_counts[blockIdx.x * BIN_SIZE + threadIdx.x] =
				bin_count[threadIdx.x];
}
;
//warning csum on all bin_counts,because BIN_SIZE=256,BLOCK_SIZE =256
template<int BIN_SIZE, int BLOCK_SIZE, int BLOCK_DIM_TILE>
__global__ void count_sum_prefixsum_op(void* __restrict__ count_bins, int k,
		 int* __restrict__ finnal_csum_buffer) {
	__shared__ int4 sm_count_bin[BIN_SIZE / 4 * 4]; //first reduce
	__shared__ int sm_count_bins[BIN_SIZE];

	int4 bin_sum;
	
	if(BLOCK_DIM_TILE<=4){
		for (int tileIdx = 0; tileIdx < BLOCK_DIM_TILE; tileIdx++) {
				int tidx = threadIdx.x + BLOCK_SIZE * tileIdx;
				int4 value = ((int4*) count_bins)[tidx];
				if (tileIdx == 0) {
					bin_sum = value;
				} else {
					bin_sum.x = bin_sum.x + value.x;
					bin_sum.y = bin_sum.y + value.y;
					bin_sum.z = bin_sum.z + value.z;
					bin_sum.w = bin_sum.w + value.w;
				}
		}
	}else{
		bin_sum.x=0;
		bin_sum.y=0;
		bin_sum.z=0;
		bin_sum.w=0;

		for (int tidx = threadIdx.x; tidx < BLOCK_DIM_TILE*BLOCK_SIZE; tidx+=BLOCK_SIZE) {
			int4 value = ((int4*) count_bins)[tidx];
				
			bin_sum.x = bin_sum.x + value.x;
			bin_sum.y = bin_sum.y + value.y;
			bin_sum.z = bin_sum.z + value.z;
			bin_sum.w = bin_sum.w + value.w;
				
		}
	}
	sm_count_bin[threadIdx.x] = bin_sum;


	__syncthreads();

	int count_bin = ((int*) sm_count_bin)[threadIdx.x];
	for (int i = 1; i < 4; i++) {
		count_bin += ((int*) sm_count_bin)[threadIdx.x + BIN_SIZE * i];
	}

	__syncthreads();

	//prefix_sum
	constexpr int log2_block_size = log2<BLOCK_SIZE>();
	CSumSync<int, 1, log2_block_size, min<5, log2_block_size>()>(&count_bin);
	sm_count_bins[threadIdx.x] = count_bin;

	__shared__ int select_bin_id[1];

	if (count_bin >= k
			&& (threadIdx.x == 0 or sm_count_bins[threadIdx.x - 1] < k)) {
		select_bin_id[0] = threadIdx.x;
	}
	__syncthreads();
	// scan all count_bins=count_bin*n  where bin_id<select_bin_id

	int select_bin = select_bin_id[0]; //todo write_out

	__shared__ int sm_reduce_sum[BLOCK_SIZE / 32];
	__shared__ int sm_reduce_sum_tile_dim[BLOCK_DIM_TILE * 4];
	if(BLOCK_DIM_TILE<=4){
		for (int tileIdx = 0; tileIdx < BLOCK_DIM_TILE; tileIdx++) {
			int tidx = threadIdx.x + BLOCK_SIZE * tileIdx;
			int4 value = ((int4*) count_bins)[tidx];
			//int blockId_at_256=tidx/64;
			int bidId0 = (tidx & 63) * 4;
			int bidId1 = (tidx & 63) * 4 + 1;
			int bidId2 = (tidx & 63) * 4 + 2;
			int bidId3 = (tidx & 63) * 4 + 3;
			int count_less_than_select_bin_id = (bidId0 <= select_bin ? value.x : 0);
			count_less_than_select_bin_id += (bidId1 <= select_bin ? value.y : 0);
			count_less_than_select_bin_id += (bidId2 <= select_bin ? value.z : 0);
			count_less_than_select_bin_id += (bidId3 <= select_bin ? value.w : 0);

			//reduce_sum on 64 threadIds
			int reduce_sum_value = count_less_than_select_bin_id;

			for (int stride = 16; stride >= 1; stride = stride >> 1) {
				int xor_value = __shfl_xor_sync(0xffffffff, reduce_sum_value,
						stride, stride * 2);
				reduce_sum_value += xor_value;
			}

			if ((threadIdx.x &31) == 0)
				sm_reduce_sum[threadIdx.x >> 5] = reduce_sum_value;
			__syncthreads();
			if (threadIdx.x < 4) {
				sm_reduce_sum[threadIdx.x * 2] +=
						sm_reduce_sum[threadIdx.x * 2 + 1];

				//send to  for csum
				sm_reduce_sum_tile_dim[tileIdx * 4 + threadIdx.x] =
						sm_reduce_sum[threadIdx.x * 2];
			}
			__syncthreads();
		}
	}else{
		for (int tidx = threadIdx.x; tidx < BLOCK_SIZE*BLOCK_DIM_TILE; tidx+=BLOCK_SIZE) {

			int4 value = ((int4*) count_bins)[tidx];
			//int blockId_at_256=tidx/64;
			int bidId0 = (tidx & 63) * 4;
			int bidId1 = (tidx & 63) * 4 + 1;
			int bidId2 = (tidx & 63) * 4 + 2;
			int bidId3 = (tidx & 63) * 4 + 3;
			int count_less_than_select_bin_id = (bidId0 <= select_bin ? value.x : 0);
			count_less_than_select_bin_id += (bidId1 <= select_bin ? value.y : 0);
			count_less_than_select_bin_id += (bidId2 <= select_bin ? value.z : 0);
			count_less_than_select_bin_id += (bidId3 <= select_bin ? value.w : 0);

			//reduce_sum on 64 threadIds
			int reduce_sum_value = count_less_than_select_bin_id;

			for (int stride = 16; stride >= 1; stride = stride >> 1) {
				int xor_value = __shfl_xor_sync(0xffffffff, reduce_sum_value,
						stride, stride * 2);
				reduce_sum_value += xor_value;
			}

			if ((threadIdx.x &31) == 0)
				sm_reduce_sum[threadIdx.x >> 5] = reduce_sum_value;
			__syncthreads();
			if (threadIdx.x < 4) {
				sm_reduce_sum[threadIdx.x * 2] +=
						sm_reduce_sum[threadIdx.x * 2 + 1];

				//send to  for csum
				sm_reduce_sum_tile_dim[(tidx>>log2<BLOCK_SIZE>()) * 4 + threadIdx.x] =
						sm_reduce_sum[threadIdx.x * 2];
			}
			__syncthreads();
		}
	}
	int csum_value = sm_reduce_sum_tile_dim[threadIdx.x % (BLOCK_DIM_TILE * 4)];
	constexpr int log2_block_size_offset = log2<BLOCK_DIM_TILE * 4>();

	CSumSync<int, 1, log2_block_size_offset, min<5, log2_block_size_offset>()>(
			&csum_value);

	if (threadIdx.x == 0) {
		finnal_csum_buffer[BLOCK_DIM_TILE * 4] = select_bin;
	}

	if (threadIdx.x < BLOCK_DIM_TILE * 4)
		finnal_csum_buffer[threadIdx.x] = csum_value;
}

template<int BLOCK_SIZE, int BLOCK_DIM_TILE>
__global__ void topk_movehead_op(void* __restrict__ hmdist, int valid_num,int* __restrict__ csum_buffer,
		int K,int* __restrict__ topkIdx) {
	int current_csum_id = blockIdx.x;
	__shared__ int global_buffer_idx[1];

	if (threadIdx.x == 0 && current_csum_id == 0) {
		global_buffer_idx[0]=0;
	} else if (threadIdx.x == 0) {
		global_buffer_idx[0]=csum_buffer[current_csum_id - 1];
	}
	__syncthreads();

	int select_bin_id = csum_buffer[gridDim.x];

	if(BLOCK_DIM_TILE>0){
		for (int tileIdx = 0; tileIdx < BLOCK_DIM_TILE; tileIdx++) {
			int tidx = threadIdx.x + BLOCK_SIZE * blockIdx.x * BLOCK_DIM_TILE
					+ BLOCK_SIZE * tileIdx;
			if (tidx < valid_num) {
				auto value = ((uint4*) hmdist)[tidx];

				// LoopCountShare<15>::write(topkIdx, global_buffer_idx,
				// 		(unsigned char*) &value, tidx, select_bin_id, K);

				update_write_global_buffer(topkIdx, global_buffer_idx,value.x, tidx*16, select_bin_id, K);
				update_write_global_buffer(topkIdx, global_buffer_idx,value.y, tidx*16+4, select_bin_id, K);
				update_write_global_buffer(topkIdx, global_buffer_idx,value.z, tidx*16+8, select_bin_id, K);
				update_write_global_buffer(topkIdx, global_buffer_idx,value.w, tidx*16+12, select_bin_id, K);

			}
		}
	}else{
		for (int tidx = threadIdx.x+blockIdx.x*BLOCK_SIZE; tidx < valid_num; tidx+=BLOCK_SIZE*gridDim.x) {
			
			
			auto value = ((uint4*) hmdist)[tidx];

			// LoopCountShare<15>::write(topkIdx, global_buffer_idx,
			// 			(unsigned char*) &value, tidx, select_bin_id, K);

			update_write_global_buffer(topkIdx, global_buffer_idx,value.x, tidx*16, select_bin_id, K);
			update_write_global_buffer(topkIdx, global_buffer_idx,value.y, tidx*16+4, select_bin_id, K);
			update_write_global_buffer(topkIdx, global_buffer_idx,value.z, tidx*16+8, select_bin_id, K);
			update_write_global_buffer(topkIdx, global_buffer_idx,value.w, tidx*16+12, select_bin_id, K);

			
		}
	}

}

#endif /* BIN_COUNT_H_ */

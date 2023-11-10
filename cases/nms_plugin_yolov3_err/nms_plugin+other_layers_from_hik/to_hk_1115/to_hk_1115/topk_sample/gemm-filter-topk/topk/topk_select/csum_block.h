/*
 * csum_block.h
 *
 *  Created on: 2021年11月9日
 *      Author: xiao.luo
 */

#ifndef CSUM_BLOCK_H_
#define CSUM_BLOCK_H_

template<int value>
__device__ constexpr  int log2(){
	for(int i=31;i>=1;i--){
		auto c=value>>i;
		if(c==1){
			return i;
		}
	}
	return 0;
}
template<int a,int b>
__device__ constexpr  int min(){
	return a>b ? b : a;
}
template<typename T,int WAY_NUM,int log2_wrap_size=5>
__device__ void CSum_wrap_inner(T* value){
	constexpr int wrap_size=1<<log2_wrap_size;
	int LaneId=threadIdx.x&(wrap_size-1);
	for(int stride=1;stride<=(wrap_size>>1);stride=stride<<1){
		T target_value[WAY_NUM];
		for(int j=0;j<WAY_NUM;j++){
			target_value[j]=__shfl_up_sync(0xffffffff,value[j],stride,wrap_size);
		}
		if((LaneId&(2*(stride-1)+1))==2*(stride-1)+1){
			for(int j=0;j<WAY_NUM;j++){
				value[j]+=target_value[j];
			}
		}
	}

	int i=1;
	for(int stride=(wrap_size>>1);stride>=2;stride=stride>>1){
        T target_value[WAY_NUM];
        for(int j=0;j<WAY_NUM;j++){
        	target_value[j]=__shfl_up_sync(0xffffffff,value[j],stride>>1,wrap_size);
        }
        if(LaneId>=stride && LaneId==(stride*(LaneId>>(log2_wrap_size-i))-1+(stride>>1))){
        	for(int j=0;j<WAY_NUM;j++){
        		value[j]+=target_value[j];
        	}
        }
        i++;
	}
}

template<typename T,int WAY_NUM,int LOG2_BLOCK_SIZE,int log2_wrap_size=5>
__device__ void CSumSync(T* ele){
	CSum_wrap_inner<T,WAY_NUM,log2_wrap_size>(ele);
	if(LOG2_BLOCK_SIZE>log2_wrap_size){
		constexpr int num=1<<(LOG2_BLOCK_SIZE-log2_wrap_size);
		__shared__ T value[num*WAY_NUM];
		if((threadIdx.x & ((1<<log2_wrap_size)-1))==((1<<log2_wrap_size)-1)){
			for(int j=0;j<WAY_NUM;j++){
				value[(threadIdx.x>>log2_wrap_size)*WAY_NUM+j]=ele[j];
			}
		}
		__syncthreads();
		T ele_sum[WAY_NUM];
		if(threadIdx.x<num){
			for(int j=0;j<WAY_NUM;j++){
				ele_sum[j]=value[threadIdx.x*WAY_NUM+j];
			}
		}
		CSum_wrap_inner<T,WAY_NUM,log2<num>()>(ele_sum);
		if(threadIdx.x<num){
			for(int j=0;j<WAY_NUM;j++){
				value[threadIdx.x*WAY_NUM+j]=ele_sum[j];
			}
		}
		__syncthreads();
		if((threadIdx.x>>5)>=1){
			for(int j=0;j<WAY_NUM;j++){
				ele[j]+=value[((threadIdx.x>>5)-1)*WAY_NUM+j];
			}
		}
	}
}

#endif /* CSUM_BLOCK_H_ */

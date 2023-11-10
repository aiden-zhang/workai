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

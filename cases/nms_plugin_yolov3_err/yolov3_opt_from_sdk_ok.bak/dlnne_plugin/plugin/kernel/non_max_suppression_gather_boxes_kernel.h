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
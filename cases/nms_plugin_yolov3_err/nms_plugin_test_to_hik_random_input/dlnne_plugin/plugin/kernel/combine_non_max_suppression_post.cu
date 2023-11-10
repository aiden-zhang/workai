
template<typename BoxType,typename ScoreType,typename SelectType,
            typename CsumType,typename IdxType>
__device__ void CombineNonMaxSuppressionPost_device_func(
                                        BoxType* boxes_buffer,
                                        int b_s0,int b_s1,int b_s2,
                                        int b_ss0,int b_ss1,int b_ss2,

										ScoreType* scores_buffer,
										int s_s0,int s_s1,int s_s2,
                                        int s_ss0,int s_ss1,int s_ss2,

										SelectType* selected_ids_buffer,
										int sel_s0,int sel_s1,int sel_s2,
                                        int sel_ss0,int sel_ss1,int sel_ss2,

										CsumType* csum_buffer,
										int csum_s0,int csum_ss0,

										IdxType* out_idxs_buffer,
										int out_idx_s0,int out_idx_s1,
										int out_idx_ss0,int out_idx_ss1,

										BoxType* out_boxes_buffer,
										int out_box_s0,
										int out_box_ss0,

										ScoreType* out_scores_buffer,
										int out_score_s0,
										int out_score_ss0,

                                        SelectType* valid_num=nullptr
										){



    int B=b_s0;
    int C=b_s1;
    int boxes_c=C;

    int count=csum_buffer[B*C-1];

	int tidx=threadIdx.x+blockIdx.x*blockDim.x;
    if(tidx==0){
        valid_num[0]=count;
    }

    int start=-1;
    int end=B*C-1;

    int current_b=0;
    int current_c=0;
    int idx=0;
    if(tidx<count){

        while(true){
            int current_idx=(start+1+end)>>1;

            int value=current_idx-1>=0?csum_buffer[current_idx-1]:-1;


            if(value <= tidx && tidx<csum_buffer[current_idx]){
                current_c=(current_idx)%C;
                current_b=(current_idx)/C;
                idx=(value>0 ? tidx-value:tidx);
                break;
            }else if(tidx>=csum_buffer[current_idx]){
                start=current_idx;
            }else if(tidx<value){
                end=current_idx;
            }
        }

        auto idx_tmp=selected_ids_buffer[current_b*sel_ss0+current_c*sel_ss1+idx*sel_ss2];

            int boxes_c_idx=(boxes_c==1?0:current_c);

            auto box_value=((float4*)boxes_buffer)[current_b*b_ss0+boxes_c_idx*b_ss1+idx_tmp*b_ss2];
            ((float4*)out_boxes_buffer)[tidx*out_box_ss0]=box_value;

            auto score_value=scores_buffer[current_b*s_ss0+current_c*s_ss1+idx_tmp*s_ss2];
            out_scores_buffer[tidx]=score_value;

            out_idxs_buffer[tidx*out_idx_ss0+0]=current_b;
            out_idxs_buffer[tidx*out_idx_ss0+1*out_idx_ss1]=current_c;

    }

}


extern "C" __global__ void fused_dl_combine_non_max_suppression_post_kernel(
    float* __restrict__ placeholder,int s0_s0,int s0_s1,int s0_s2,int s0_ss0,int s0_ss1,int s0_ss2,
     float* __restrict__ placeholder1,int s1_s0,int s1_s1,int s1_s2,int s1_ss0,int s1_ss1,int s1_ss2,
      int* __restrict__ placeholder2,int s2_s0,int s2_s1,int s2_s2,int s2_ss0,int s2_ss1,int s2_ss2,
      int* __restrict__ placeholder4,int s3_s0,int s3_ss0,

       int* __restrict__ combine_non_max_suppression_post_op_v0,int o0_s0,int o0_s1,int o0_ss0,int o0_ss1,
       float* __restrict__ combine_non_max_suppression_post_op_v1,int o1_s0,int o1_s1,int o1_ss0,int o1_ss1,
        float* __restrict__ combine_non_max_suppression_post_op_v2,int o2_s0,int o2_ss0,
         int* __restrict__ combine_non_max_suppression_post_op_v3) {

   CombineNonMaxSuppressionPost_device_func<float,float,int,int,int>
  (placeholder,s0_s0,s0_s1,s0_s2,s0_ss0,s0_ss1,s0_ss2,
   placeholder1,s1_s0,s1_s1,s1_s2,s1_ss0,s1_ss1,s1_ss2,
   placeholder2,s2_s0,s2_s1,s2_s2,s2_ss0,s2_ss1,s2_ss2,

   placeholder4,s3_s0,s3_ss0,
   combine_non_max_suppression_post_op_v0, o0_s0,o0_s1,o0_ss0,o0_ss1,
   combine_non_max_suppression_post_op_v1, o1_s0,o1_ss0,
   combine_non_max_suppression_post_op_v2, o2_s0,o2_ss0,
   combine_non_max_suppression_post_op_v3);
}

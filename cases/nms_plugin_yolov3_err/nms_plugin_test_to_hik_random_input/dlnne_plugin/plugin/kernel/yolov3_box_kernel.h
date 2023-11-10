#include <cuda_runtime.h>

#define MIN(a,b) ((a)<(b) ? (a):(b))

template<typename T0>
__device__ auto sigmoid(T0 t0)
{
	return 1/(1 + expf(-t0));
}

template<typename T0>
__device__ auto rcp(T0 t0)
{
	return 1.0f/t0;
}

__device__ auto BoxesGetFeats(const float4* feats_buffer,
															const float* anchors_buffer,
															const int* input_shape_buffer,
															const int tidx_,
															const int b_idx, const int h_idx, const int w_idx,const int anchors_idx,
															const int H,const int W)
{
	//stage1: BoxesGetFeats
	float4 out_value;

	int grid_h = H;
	int grid_w = W;
	float4 value = feats_buffer[tidx_]; 

	out_value.x=(sigmoid(value.y)+h_idx) * rcp(grid_h*1.0f);
	out_value.y=(sigmoid(value.x)+w_idx) * rcp(grid_w*1.0f);

	out_value.z=expf(value.w); 
	out_value.z=out_value.z*anchors_buffer[anchors_idx*2 + 1]; //load anchors'h
	out_value.z=out_value.z/input_shape_buffer[1];//h,w so load 0

	
	out_value.w=expf(value.z); //thread2, load(x,y,w,h)-->h
	out_value.w=out_value.w*anchors_buffer[anchors_idx*2]; //load anchors'h
	out_value.w=out_value.w/input_shape_buffer[2];//h,w so load 0

	return out_value;
}

__device__ auto CorrectBoxes(float4 box_value,
														 const int* image_shape_buffer,
														 const int* input_shape_buffer,
														 const int b_idx)
{
	float4 out_value=box_value;

	auto image_shape_h=image_shape_buffer[b_idx*2];
	auto image_shape_w=image_shape_buffer[b_idx*2 + 1];

	auto input_h=input_shape_buffer[1];
	auto input_w=input_shape_buffer[2];

	auto scale=MIN(input_h*1.0f/image_shape_h,input_w*1.0f/image_shape_w);
	int new_shape_h=image_shape_h*scale+0.5f;
	int new_shape_w=image_shape_w*scale+0.5f;
	auto offset_h=(input_h-new_shape_h)/2.0f/input_h;
	auto offset_w=(input_w-new_shape_w)/2.0f/input_w;
	
	auto scale_h=input_h*1.0f/new_shape_h;
	auto scale_w=input_w*1.0f/new_shape_w;



	box_value.x = (box_value.x - offset_h) * scale_h;
				
	auto box_value_h = box_value.z*scale_h; 
	
	out_value.x = (box_value.x - box_value_h/2)*image_shape_h;
	out_value.z = (box_value.x + box_value_h/2)*image_shape_h;

	//thread1 compute x1
	box_value.y = (box_value.y - offset_w) * scale_w;
	
	auto box_value_w = box_value.w*scale_w; 
	
	out_value.y = (box_value.y - box_value_w/2)*image_shape_w;
	out_value.w = (box_value.y + box_value_w/2)*image_shape_w;

	return out_value;
}

__device__ void dl_boxes_float4_device_func(
		const float4* feats_buffer, 
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	for(int tidx_ = threadIdx.x + blockDim.x * blockIdx.x; tidx_ < B*H*W*anchor_num; tidx_ += blockDim.x*gridDim.x)
	{
		int tidx = tidx_*4;
		auto box_class_idx= 0; //tidx % 4; // 1,2 is x,y  3,4 is w,h
		int prev_idx=tidx / 4; //which (x,y,w,h), total is H*W*anchor_num every batch
		auto out_num_idx=prev_idx%(anchor_num*H*W);//this batch's which (x,y,w,h)
		auto b_idx=prev_idx/(anchor_num*H*W);//which batch

		auto anchor_idx=out_num_idx%anchor_num; //total(H*W*anchor_num)'s which anchor
		auto w_idx=(out_num_idx/anchor_num)%W; //total(H*W)'s which W
		auto h_idx=(out_num_idx/anchor_num)/W;

		float4 out_value = BoxesGetFeats(feats_buffer,
																		 anchors_buffer,
																		 input_shape_buffer,
																		 tidx_,
																		 b_idx,h_idx,w_idx,anchor_idx,
																		 H,W);
		out_value = CorrectBoxes(out_value,
														 image_shape_buffer,
														 input_shape_buffer,
														 b_idx);

		boxes_buffer[tidx_] = out_value;

	}
}

extern "C" __global__  void dl_boxes_float4_device_func_global(
		const float4* feats_buffer, 
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func(
		feats_buffer, 
		anchors_buffer,//[3,2]
		input_shape_buffer, //shape == (4,) nhwc || nchw
		image_shape_buffer, //shape == (none,2) hw
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}

__device__  void dl_boxes_float4_device_func_kernel(
		const float4* feats_buffer, 
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func(
		feats_buffer, 
		anchors_buffer,//[3,2]
		input_shape_buffer, //shape == (4,) nhwc || nchw
		image_shape_buffer, //shape == (none,2) hw
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}









__device__ auto BoxesGetFeatsAligned(const float4* feats_buffer,
															const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
															const float* anchors_buffer,
															const int* input_shape_buffer,
															const int tidx_,
															const int b_idx, const int h_idx, const int w_idx,const int anchors_idx,
															const int H,const int W)
{
	//stage1: BoxesGetFeats
	int f4_feats_s3 = feats_s3;
	int f4_feats_s2 = feats_s2/4;
	int f4_feats_s1 = feats_s1/4;
	int f4_feats_s0 = feats_s0/4;

	float4 out_value;

	int grid_h = H;
	int grid_w = W;
	// float4 value = feats_buffer[tidx_]; //(n,h,w,12) --> (n,h,w,3,4) --> (float4)(n,h,w,3) -->(float4 aligned)(n,h,w,4)
	float4 value = feats_buffer[b_idx*H*W*4 + h_idx*W*4 + w_idx*4 + anchors_idx];
	// float4 value = feats_buffer[b_idx*f4_feats_s0 + h_idx*f4_feats_s1 + w_idx*f4_feats_s2 + anchors_idx*f4_feats_s3];

	out_value.x=(sigmoid(value.y)+h_idx) * rcp(grid_h*1.0f);
	out_value.y=(sigmoid(value.x)+w_idx) * rcp(grid_w*1.0f);

	out_value.z=expf(value.w); //thread2, load(x,y,w,h)-->h
	out_value.z=out_value.z*anchors_buffer[anchors_idx*2 + 1]; //load anchors'h
	out_value.z=out_value.z/input_shape_buffer[1];//h,w so load 0

	
	out_value.w=expf(value.z); //thread2, load(x,y,w,h)-->h
	out_value.w=out_value.w*anchors_buffer[anchors_idx*2]; //load anchors'h
	out_value.w=out_value.w/input_shape_buffer[2];//h,w so load 0

	return out_value;
}

__device__ void dl_boxes_float4_device_func_aligned(
		const float4* feats_buffer, //typename Tensor0::DATA_TYPE* feats_buffer,//[None,h,w,12]
		const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	for(int tidx_ = threadIdx.x + blockDim.x * blockIdx.x; tidx_ < B*H*W*anchor_num; tidx_ += blockDim.x*gridDim.x)
	{
		int tidx = tidx_*4;
		auto box_class_idx= 0; //tidx % 4; // 1,2 is x,y  3,4 is w,h
		int prev_idx=tidx / 4; //which (x,y,w,h), total is H*W*anchor_num every batch
		auto out_num_idx=prev_idx%(anchor_num*H*W);//this batch's which (x,y,w,h)
		auto b_idx=prev_idx/(anchor_num*H*W);//which batch

		auto anchor_idx=out_num_idx%anchor_num; //total(H*W*anchor_num)'s which anchor
		auto w_idx=(out_num_idx/anchor_num)%W; //total(H*W)'s which W
		auto h_idx=(out_num_idx/anchor_num)/W;

		float4 out_value = BoxesGetFeatsAligned(feats_buffer,
																		 feats_s0,feats_s1,feats_s2,feats_s3,
																		 anchors_buffer,
																		 input_shape_buffer,
																		 tidx_,
																		 b_idx,h_idx,w_idx,anchor_idx,
																		 H,W);
		out_value = CorrectBoxes(out_value,
														 image_shape_buffer,
														 input_shape_buffer,
														 b_idx);

		boxes_buffer[tidx_] = out_value;

	}
}

extern "C" __global__  void dl_boxes_float4_device_func_global_aligned(
		const float4* feats_buffer, //typename Tensor0::DATA_TYPE* feats_buffer,//[None,h,w,12]
		const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
		const float* anchors_buffer,//[3,2]
		const int* input_shape_buffer, //shape == (4,) nhwc || nchw
		const int* image_shape_buffer, //shape == (none,2) hw
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func_aligned(
		feats_buffer, //typename Tensor0::DATA_TYPE* feats_buffer,//[None,h,w,12]
		feats_s0,feats_s1,feats_s2,feats_s3,
		anchors_buffer,//[3,2]
		input_shape_buffer, //shape == (4,) nhwc || nchw
		image_shape_buffer, //shape == (none,2) hw
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}

__device__  void dl_boxes_float4_device_func_kernel_aligned(
		const float4* feats_buffer, 
		const int feats_s0,const int feats_s1,const int feats_s2,const int feats_s3,
		const float* anchors_buffer,
		const int* input_shape_buffer, 
		const int* image_shape_buffer,
		const int B, const int H, const int W, const int anchor_num, const int class_num,
		float4* boxes_buffer
		)
{
	dl_boxes_float4_device_func_aligned(
		feats_buffer, 
		feats_s0,feats_s1,feats_s2,feats_s3,
		anchors_buffer,
		input_shape_buffer, 
		image_shape_buffer, 
		B, H, W, anchor_num, class_num,
		boxes_buffer
		);
}

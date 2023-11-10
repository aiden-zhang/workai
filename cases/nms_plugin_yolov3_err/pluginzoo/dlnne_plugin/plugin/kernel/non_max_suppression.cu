#include <cuda_runtime.h>


#define MAX(a,b) ((a)>(b) ? (a):(b))
#define MIN(a,b) ((a)<(b) ? (a):(b))

template<typename T>
struct BaseTypeFromVector
{
  typedef T type;
};
template<>
struct BaseTypeFromVector<float4>
{
  typedef float type;
};

template<typename T,int center_point_box>
class Box{
public:
  using BaseT =typename BaseTypeFromVector<T>::type;
  __device__  Box(T a)
  {
    this->y0=MIN(a.x,a.z);
    this->x0=MIN(a.y,a.w);
    this->y1=MAX(a.x,a.z);
    this->x1=MAX(a.y,a.w);
  }

  __device__ BaseT area(){
    return (y1-y0)*(x1-x0);

  }
  BaseT y0;
  BaseT x0;
  BaseT y1;
  BaseT x1;
};

template<typename T>
class Box<T,1>
{
public:
  using BaseT =typename BaseTypeFromVector<T>::type;

  __device__  Box(T a){
    BaseT x_center=a.x;
    BaseT y_center=a.y;
    this->width=a.z;
    this->height=a.w;

    this->y0=y_center-height/2;
    this->x0=x_center-width/2;
    this->y1=y0+height;
    this->x1=x0+width;

  }

  __device__ BaseT area(){
    return height*width;
  }
  BaseT y0;
  BaseT x0;
  BaseT y1;
  BaseT x1;
  BaseT height;
  BaseT width;
};

template<typename BOX,typename T>
__device__ bool IOU_GE(BOX a,BOX b,T iou_threshold)
{
  // T y_diff=MAX(0,MIN(a.y1,b.y1)-MAX(a.y0,b.y0));
  // T x_diff=MAX(0,MIN(a.x1,b.x1)-MAX(a.x0,b.x0));

  // return y_diff*x_diff>iou_threshold*(a.area()+b.area()-y_diff*x_diff);
  T ymin_i = MIN(a.y0,a.y1);
  T xmin_i = MIN(a.x0,a.x1);
  T ymax_i = MAX(a.y0,a.y1);
  T xmax_i = MAX(a.x0,a.x1);

  T ymin_j = MIN(b.y0,b.y1);
  T xmin_j = MIN(b.x0,b.x1);
  T ymax_j = MAX(b.y0,b.y1);
  T xmax_j = MAX(b.x0,b.x1);

  T area_i = (ymax_i - ymin_i)*(xmax_i - xmin_i);
  T area_j = (ymax_j - ymin_j)*(xmax_j - xmin_j);

  if(area_i <=0 || area_j <= 0) return false;

  T interection_ymin = MAX(ymin_i,ymin_j);
  T interection_xmin = MAX(xmin_i,xmin_j);
  T interection_ymax = MIN(ymax_i,ymax_j);
  T interection_xmax = MIN(xmax_i,xmax_j);

  T interection_area = MAX((interection_ymax-interection_ymin),0)*
                        MAX((interection_xmax-interection_xmin),0);

  return interection_area>iou_threshold*(area_i+area_j-interection_area);
}

template<typename OP0,
         typename OP1,
         typename OP2,
         typename OP3,
         typename OP4,
         typename OP5,
         typename TensorBool,
         typename TensorIdx,
         typename TensorCount,
         int center_point_box=1>
class NonMaxSuppressionOp
{
public:
  using BOX = Box<float4, center_point_box>;

  __device__ NonMaxSuppressionOp(OP0 boxes, //tensor ptr
                                 OP1 scores, //tensor ptr
                                 OP2 max_output_boxes_per_class, //scalar
                                 OP3 iou_threshold, //scalar
                                 OP4 score_threshold, //scalar
                                 OP5 sort_size, //scalar
                                 TensorBool disable, //tensor ptr
                                 TensorIdx output_ids, //tensor ptr
                                 TensorCount count, //ptr one num 
                                 int* socres_dim_sizes
                                 )
  {
    this->boxes=boxes;
    this->scores=scores;
    this->max_output_boxes_per_class=max_output_boxes_per_class;
    this->iou_threshold=iou_threshold;
    this->score_threshold=score_threshold;
    this->sort_size=sort_size;

    this->disable=disable;
    this->output_ids=output_ids;
    this->count=count;

    __shared__ float4 current_box[1];
    __shared__ bool is_finshed[1];
    __shared__ int stride[2];
    
    this->current_box=current_box;
    this->is_finshed=is_finshed;

    this->stride=stride;

    this->socres_dim_sizes = socres_dim_sizes;
  }

  template<typename IndexHelper> 
  __device__ void AutoLoad(IndexHelper idx)
  {
    
    
    auto iou_threshold = this->iou_threshold;
    float score_threshold;
    
    int max_output_boxes;

    int count=0;
    if(threadIdx.x==0)
    {
      
      score_threshold = (float)(this->score_threshold);
      
      max_output_boxes = this->max_output_boxes_per_class;

      this->stride[0] = 0;
      if(max_output_boxes == 0)
      {
        max_output_boxes=-1;
      }
      int start_idx = 0;
      
      int end_idx = socres_dim_sizes[0] - 1;
      
      if(sort_size >= 0)
      {
        end_idx = sort_size - 1;
      }
      while(start_idx+1 >= end_idx && start_idx != end_idx)
      {
              int center_idx = (start_idx + end_idx + 1)>>1;
              
              auto current_score = scores[center_idx];
              
              if(current_score < score_threshold)
              {
                  end_idx=center_idx;
              }
              else
              {
                  start_idx=center_idx;
              }
       }
       this->stride[1] = end_idx;//find last score
    }
    

    
    int iter_idx=0;
    do{
        __syncthreads();
      if(threadIdx.x == 0)
      {
        this->is_finshed[0] = true;

        for(int stride=this->stride[0]; stride <= this->stride[1]; stride++)
        {
          
          auto start_idx = stride;
          
          bool isDisable;
          if(iter_idx == 0)
          {
            isDisable = false;  
          }
          else
          {
           
            isDisable = disable[start_idx];  
          }
          
          
          auto score = scores[start_idx];
          
          if(score <= score_threshold || count == max_output_boxes)
          {
            break;
          }
          
          if(isDisable == false)
          {
            
            this->current_box[0] = boxes[start_idx];

            
            output_ids[count] = stride;
            count+=1;

            this->stride[0] = stride+1;
            if(this->stride[0] > this->stride[1])
            {
              this->is_finshed[0]=true;
            }
            else
            {
              this->is_finshed[0]=false;
            }
            break;
          }
        }
      }
      __threadfence();
      __syncthreads();
      
      if(this->is_finshed[0])
      {
          if(threadIdx.x==0)
          {
              
            this->count[0] = count;
            
          }
        return;
      }
      BOX current_box_selected(this->current_box[0]);

      for(int stride=this->stride[0]+threadIdx.x; stride <= this->stride[1]; stride+=blockDim.x)
      {
        
        auto start_idx = stride;

        
        BOX box(boxes[start_idx]);
        if(iter_idx == 0)
        {
          
          disable[start_idx] = false; 
        }
        
        bool is_disable = disable[start_idx];
       
        if(!is_disable && IOU_GE(current_box_selected,box,iou_threshold))
        {
          
          disable[start_idx] = true;
        }
        
      }
      iter_idx++;
      __threadfence();
      __syncthreads();
    

    }while(this->stride[0]<=this->stride[1]);


  }
private:
  OP0 boxes;//[sptial_dimension,4]
  OP1 scores;//[sptial_dimension]
  OP2 max_output_boxes_per_class;//[1]
  OP3 iou_threshold;//[1]
  OP4 score_threshold;//[1]
  OP5 sort_size;//[1]

  TensorBool disable;
  TensorIdx output_ids;
  TensorCount count;

  
  float4* current_box=nullptr;
  bool* is_finshed=nullptr;
  int* stride=nullptr;

  int* socres_dim_sizes;
};


template<typename OP0,
         typename OP1,
         typename OP2,
         typename OP3,
         typename OP4,
         typename OP5,
         typename TensorBool,
         typename TensorIdx,
         typename TensorCount,
         int center_point_box=true>
__device__ auto NonMaxSuppression(
                                  OP0 boxes, //tensor
                                  OP1 scores, //tensor
                                  OP2 max_output_boxes_per_class, //scalar
                                  OP3 iou_threshold, //scalar
                                  OP4 score_threshold,//scalar
                                  OP5 op5,//scalar
                                  TensorBool disable, //tensor
                                  TensorIdx output_ids,//tensor
                                  TensorCount count, //ptr one num 
                                  int* socres_dim_sizes
                                  )
{
  return NonMaxSuppressionOp<OP0,OP1,OP2,OP3,OP4,OP5,TensorBool,TensorIdx,TensorCount,center_point_box>
        (boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold,op5,disable,output_ids,count,socres_dim_sizes);
}

template<int center_point_box=0,int sorted=1>
__device__  void NonMaxSuppression_device_func(
                               float4* boxes_buffer,
                               float* scores_buffer,
                               int* max_output_boxes_per_class_buffer,
                               float* iou_threshold_buffer,
                               float* scores_threshold_buffer,
                               int box_class_num, //80
                               int B, //1
                               int C, //80
                               int box_s, 
                               int scores_s, 
                              bool* is_disable_buffer,
                              int* boxIds_buffer,
                              int* count_buffer,
                               int* sort_size_buffer=nullptr)
{
  int scores_dim_sizes[3]={B,C,scores_s}; 
  int box_dim_sizes[3]={B,box_class_num,box_s}; 

  
   float iou_threshold = iou_threshold_buffer[0];
   float scores_threshold = scores_threshold_buffer[0];

  
  
  int count_dim_sizes[2]={B,C};//(1,80)
  
  int batchIdx = blockIdx.x / C;
  
  int classIdx = blockIdx.x % C;

  int box_class_idx=(box_class_num==1?0:classIdx);

 
   float4& boxes_sub_buffer = boxes_buffer[(batchIdx*box_class_num*box_s + box_class_idx*box_s + 0)]; //{B,box_class_num,box_s}
  
   float& scores_sub_buffer = scores_buffer[(batchIdx*C*scores_s + classIdx*scores_s + 0)]; //{B,C,scores_s}
  
  
   int& max_output_boxes_per_class_sub_buffer = max_output_boxes_per_class_buffer[0]; 

  

 
  int& boxIds_sub_buffer = boxIds_buffer[(batchIdx*box_class_num*box_s + classIdx*box_s + 0)]; //{B,box_class_num,box_s}
  
  bool& is_disable_sub_buffer = is_disable_buffer[(batchIdx*box_class_num*box_s,classIdx*box_s + 0)]; //{B,box_class_num,box_s}
 
  int& count_sub_buffer=count_buffer[(batchIdx*C + classIdx)]; //{B,C}


  
  int boxes_dim_sizes[1] = {box_s}; //{B,box_class_num,box_s}
 
  int socres_dim_sizes[1] = {scores_s}; //{B,C,scores_s}


 
   float4* boxes_sub = &boxes_sub_buffer;
 
   float* scores_sub = &scores_sub_buffer;
  

  
   int max_output_boxes_per_class_sub = max_output_boxes_per_class_sub_buffer; 
  

  
  int* boxIds_sub = &boxIds_sub_buffer;
 
  bool* is_disable_sub = &is_disable_sub_buffer;
  
  int* count = &count_sub_buffer;
  int sort_size=-1;
  if(sort_size_buffer!=nullptr)
    sort_size = sort_size_buffer[batchIdx*C+classIdx];

  auto non = NonMaxSuppression<
                                decltype(boxes_sub),
                                decltype(scores_sub),
                                decltype(max_output_boxes_per_class_sub),
                                decltype(iou_threshold),
                                decltype(scores_threshold),
                                decltype(sort_size),
                                decltype(is_disable_sub),
                                decltype(boxIds_sub),
                                decltype(count),
                                center_point_box>(boxes_sub, //tensor
                                                  scores_sub, //tensor
                                                  max_output_boxes_per_class_sub, //scalar
                                                  iou_threshold, //scaler
                                                  scores_threshold, //scalar
                                                  sort_size, //scalar
                                                  is_disable_sub, //tensor
                                                  boxIds_sub, //tensor
                                                  count, //ptr one num
                                                  socres_dim_sizes);

  non.template AutoLoad(threadIdx.x);
}

extern "C" __global__  void NonMaxSuppression_device_func_global(
                               float4* boxes_buffer,
                               float* scores_buffer,
                               int* max_output_boxes_per_class_buffer,
                               float* iou_threshold_buffer,
                               float* scores_threshold_buffer,
                               int box_class_num, int B, int C,
                               int box_s,
                               int scores_s,
                              bool* is_disable_buffer,
                              int* boxIds_buffer,
                              int* count_buffer,
                               int* sort_size_buffer=nullptr)
{
  NonMaxSuppression_device_func<0,1>(boxes_buffer,
                               scores_buffer,
                                max_output_boxes_per_class_buffer,
                               iou_threshold_buffer,
                               scores_threshold_buffer,
                               box_class_num, B, C,
                               box_s,
                               scores_s,
                               is_disable_buffer,
                               boxIds_buffer,
                               count_buffer,
                               sort_size_buffer);
}
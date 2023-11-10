#include <cassert>
#include <stdlib.h>
#include <numeric>
#include <sstream>
#include <dlfcn.h>
#include "cuda_runtime_api.h"
#include "Detector.h"
#include "util.h"
#include "image_proc.h"

// #include "../../../pluginzoo/NMS/tvm_op/plugin_register.h"

#ifdef DL_OPENCV_ENABLE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#endif

using namespace dl::nne;
using namespace  std; 

#define MAX_BATCHSIZE  32

inline int volume(const Dims &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}
inline uint32_t getElementSize(dl::nne::DataType type) {
  switch (type) {
    case dl::nne::DataType::kINT64:
    case dl::nne::DataType::kUINT64:
      return 8;
    case dl::nne::DataType::kINT32:
    case dl::nne::DataType::kUINT32:
    case dl::nne::DataType::kFLOAT32:
      return 4;
    case dl::nne::DataType::kINT16:
    case dl::nne::DataType::kUINT16:
    case dl::nne::DataType::kFLOAT16:
      return 2;
    case dl::nne::DataType::kINT8:
    case dl::nne::DataType::kUINT8:
      return 1;
    case dl::nne::DataType::kUNKNOWN_TYPE:
      return 0;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

DlDetector::DlDetector()
{
    bindings.clear();
    buffer_datas.clear();
    buffer_sizes.clear();
    // initPluginRegister();
}

DlDetector::~DlDetector()
{
    for(int i = 0; i < bindings.size(); i++){
        if(NULL != bindings[i]){
            cudaFree(bindings[i]);            
        }
    }
    bindings.clear();

    for(int i = 0; i < buffer_datas.size(); i++){
        if(NULL != buffer_datas[i]){
            cudaFreeHost(buffer_datas[i]);            
        }
    }
    buffer_datas.clear();


    printf("---free  network \n");
    if(network)
    {
        network->Destroy();
        network=NULL;
    }

    printf("---free  context \n");
    if(context)
    {
        context->Destroy();
        context=NULL;
    }
    printf("---free  engine \n");
    if(engine)
    {
        engine->Destroy();
        engine=NULL;
    }
    printf("---free  builder \n");
    if(builder)
    {
        builder->Destroy();
        builder =NULL;
    }
    cudaStreamDestroy(m_Stream);
}
void DlDetector::LoadModel( const std::string& modelFile, int dlgpu_id, int maxBatch )
{
    if(dlgpu_id >= 0){
        m_dlGpuId = dlgpu_id;
    }else{
        m_dlGpuId = 0;
    }
    if(maxBatch > 0 && maxBatch < 1024)
    {
        m_iMaxBatch = maxBatch;
    }
    else{
        m_iMaxBatch = MAX_BATCHSIZE;
    }

    std::string slz_file_name = modelFile + ".engine";
    cudaDeviceProp prop;
    cudaSetDevice(m_dlGpuId);
    cudaGetDeviceProperties(&prop, m_dlGpuId);
    BuilderConfig             builderCfg;
    ClusterConfig             clusterCfg;

    int cluster_count = prop.clusterCount;  //gpu核数
    builderCfg.callback = NULL;             //子图划分参数
    builderCfg.dump_dot = false;            //生成dot    
    builderCfg.dump_ir = false;             //生成ir    
    builderCfg.max_batch_size =  m_iMaxBatch;
    builderCfg.print_profiling = false;     //打印pro 日志

    if(2 == cluster_count){
        builderCfg.ws_mode = kShare2;
        clusterCfg = kCluster01;
    }else if (1 == cluster_count){
        builderCfg.ws_mode = kSingle;;
        clusterCfg = kCluster0;
    }else if (4 == cluster_count){
        builderCfg.ws_mode = kShare4;
        clusterCfg = kCluster0123;
    }

    std::ifstream serializefile(slz_file_name, std::ios::binary);

    std::stringstream current_file_stream;
    current_file_stream << get_current_dir_name();
    std::string ssd_opt_so_path = current_file_stream.str() + std::string("/../../pluginzoo/dlnne_plugin_build/libyolov3_opt_plugin.so");
    void *so_handle = nullptr;
    so_handle = dlopen(ssd_opt_so_path.c_str(), RTLD_NOW);
    if(!so_handle){
        std::cout<<"Load libDetectionOutput_opt_plugin failed"<<std::endl;
        exit(-1);
    }


    if (!serializefile.good())//无序列化文件
	{      
        bool result;
        printf("---CreateInferBuilder \n");
        builder = CreateInferBuilder();
        ASSERT(builder != nullptr);

        printf("---CreateNetwork \n");
        network = builder->CreateNetwork();
        ASSERT(network != nullptr);

        printf("---SetBuilderConfig \n");
        builder->SetBuilderConfig(builderCfg);

        printf("---CreateParser \n");
        Parser *parser = CreateParser();
        ASSERT(parser != nullptr);

        printf("---RegisterInput \n");

        string ssd_opt_kernel_path = current_file_stream.str() + std::string("/../../pluginzoo/dlnne_plugin/plugin/kernel");
        setenv("YOLOV3_PLUGIN_KERNEL_PATH", ssd_opt_kernel_path.c_str(), 1); // NORMALIZE_PLUGIN_KERNEL_PATH

        
        std::string front_end_path  = current_file_stream.str() + std::string("/../../pluginzoo/front_end.py");
        std::string tvm_so_path     = current_file_stream.str() + std::string("/../../pluginzoo/dlnne_plugin_build/libyolov3_opt_tvm.so");
        parser->RegisterUserOp(tvm_so_path.c_str(), front_end_path.c_str(), "custom_op");

        printf("---Parse \n");
        result = parser->Parse(modelFile.c_str(), *network);
        ASSERT(result);

        printf("---BuildEngine \n");
        engine = builder->BuildEngine(*network);
        ASSERT(engine != nullptr);

        printf("---Serialize model \n");
        HostMemory* serializeModel = engine->Serialize();
        ofstream ofs(slz_file_name);
        ofs.write((char *)serializeModel->Data(), serializeModel->Size());
        ofs.close();
        serializeModel->Destroy(); 
        parser->Destroy();
    }else{//有序列化文件
        char* slz_data_ptr = nullptr; 

        serializefile.seekg(0, serializefile.end);
        size_t data_size = serializefile.tellg();
        serializefile.seekg(0, serializefile.beg);
        printf("---seriallize file size = %d \n", data_size);
        cudaMallocHost(reinterpret_cast<void **>(&slz_data_ptr), data_size);
        serializefile.read(slz_data_ptr, data_size);
        engine = Deserialize(slz_data_ptr, data_size);
        serializefile.close();
        cudaFreeHost(slz_data_ptr);
    }

    printf("---CreateExecutionContext \n");
    context = engine->CreateExecutionContext(clusterCfg);
    ASSERT(context != nullptr);

    cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking);

    printf("---allocate buffers \n");

    for (int i = 0; i < engine->GetNbBindings(); ++i) {
        void *ptr = nullptr;
        float *data_ptr = nullptr;
        auto dims = engine->GetBindingDimensions(i);
        auto type = engine->GetBindingDataType(i);
        int vol = volume(dims) * m_iMaxBatch;
        size_t size = vol * getElementSize(type);

        //cout<<"GetNbBindings:" << i << " size:" << size << endl;

        cudaMalloc(&ptr, size);
        buffer_sizes.push_back(size);
        bindings.push_back(ptr);

        // if (!engine->BindingIsInput(i))
        {
            cudaMallocHost(reinterpret_cast<void **>(&data_ptr), size);
            buffer_datas.push_back(data_ptr);
        }
    }
    printf("---init success \n");
}
typedef  struct 
{
    float fRatio;
    int pad_x;
    int pad_y;
} DL_Resize_Param_ST;

// #define INPUT_BGR_IMG
int DlDetector::Detect(vector<DlImage*> Src, std::vector< std::vector<DL_Detect_Result_ST> > &_Results)
{ 
    int in_id = -1;
    for (int i = 0; i < engine->GetNbBindings(); ++i) 
    {
        if (engine->BindingIsInput(i)) //输出
        { 
            in_id = i;
            break;
        }
    }
    auto dims = engine->GetBindingDimensions(in_id);

    char* fDest = (char*)(bindings[in_id]);

    int batch = Src.size();

    if(batch > m_iMaxBatch )
    {
        printf("input batch size is error. max batch %d input batch %d\n", m_iMaxBatch, batch);
        abort();
    }

    int  channel = dims.d[1];
    int  input_h = dims.d[2];
    int  input_w = dims.d[3];  

    int in_dev_offset = 0;

    std::vector<DL_Resize_Param_ST> stResizeParamSet;
    stResizeParamSet.clear();
    for(int i = 0; i < batch; i++){

        if (Src[i]->iChannel != 3 ) {
			printf("[Error] Not support channel or stride. channel: %d\n", Src[i]->iChannel);
			abort();
		}

        int src_w = Src[i]->iWidth, src_h = Src[i]->iHeight;
        float B_mean = 0, G_mean = 0, R_mean = 0;
		float scale = 1.f;
        float B_std = 1.0/255, G_std = 1.0/255, R_std = 1.0/255;
		
        int img_w, img_h, pad_w, pad_h;
        float r_w = input_w / (src_w*1.0);
        float r_h = input_h / (src_h*1.0);
        if (r_h > r_w) {//宽大
            img_w = input_w;
            img_h = r_w * src_h;
            pad_w = 0;
            pad_h = (input_h - img_h) / 2;

            DL_Resize_Param_ST stResizeParam = {r_w, pad_w, pad_h};

            stResizeParamSet.push_back(stResizeParam);
        } else {
            img_w = r_h * src_w;
            img_h = input_h;
            pad_w = (input_w - img_w) / 2;
            pad_h = 0;

            DL_Resize_Param_ST stResizeParam = {r_h, pad_w, pad_h};

            stResizeParamSet.push_back(stResizeParam);
        }
#ifdef INPUT_BGR_IMG // input RGB or BGR images 
		RGBROIBilinearResizeNormPadPlane((uint8_t*)Src[i]->data, (float *)(fDest + i * input_w * input_h * channel * sizeof(float)),
			src_w, src_h, input_w, input_h,
            img_w, img_h, pad_w, pad_h,
			0, 0, src_w, src_h,
			scale,
			B_mean, G_mean, R_mean,
            B_std, G_std, R_std,
            0, 0, 0, 
            true,
			m_Stream);
#else //yuv imgaes
         /* 预处理核函数 (pixel_value* scale - mean) * std  上下两侧pading */
        YU12ToRGBBilinearResizeNormPlane((uint8_t*)Src[i]->data, (float *)(fDest + i * input_w * input_h * channel * sizeof(float)), 
        	src_w, src_h,
            img_w, img_h,
            input_w, input_h,
            pad_w, pad_h, 
            B_mean, G_mean, R_mean,           //mean
            B_std, G_std, R_std,     //std
            scale,               //scale
            0.0, 0.0, 0.0,     //pad
            m_Stream);
#endif
#ifdef DL_OPENCV_ENABLE
        {//检查输入图像
            cv::Mat img_yuv = cv::Mat(src_h*3/2, src_w, CV_8UC1);
            cudaMemcpy((char*)(img_yuv.data), (char*)(Src[i]->data), src_w*src_h*3/2,cudaMemcpyDeviceToHost);
            cv::Mat gbr24;
            cvtColor(img_yuv,gbr24, cv::COLOR_YUV2BGR_I420);
            cv::imwrite("image.jpg", gbr24);        
        }
        {//检测预处理后的图像
            float* data = (float*)malloc(640*640*4*3);
            cudaMemcpy((char*)data, (char*)((fDest + i * input_w * input_h * channel * sizeof(float))), 640*640*4*3,cudaMemcpyDeviceToHost);
            for(int k = 0; k < 640*640*3; k++)
            {
                 data[k] = data[k] * 255;
            }
            cv::Mat f32ImgC1 = cv::Mat(640, 640, CV_32FC1, data);
            cv::Mat f32ImgC2 = cv::Mat(640, 640, CV_32FC1, data + 640*640);
            cv::Mat f32ImgC3 = cv::Mat(640, 640, CV_32FC1, data + 640*640*2);
            vector<cv::Mat> mv;
            mv.push_back(f32ImgC1);
            mv.push_back(f32ImgC2);
            mv.push_back(f32ImgC3);
            cv::Mat fImge;
            cv::merge(mv, fImge);
            cv::Mat uImg;
            free(data);
            fImge.convertTo(uImg, CV_8UC3);
            cv::imwrite("image1.jpg", uImg);        
        }
#endif
    }    
    // cudaStreamSynchronize(m_Stream);
    context->Enqueue(batch, bindings.data(), m_Stream, nullptr);
    // cudaStreamSynchronize(m_Stream);
    for (int i = 0; i < engine->GetNbBindings(); ++i) 
    {
        if (!engine->BindingIsInput(i)) //输出
        { 
            cudaMemcpyAsync(buffer_datas[i], bindings[i], buffer_sizes[i],cudaMemcpyDeviceToHost, m_Stream);
        }
    }
    cudaStreamSynchronize(m_Stream);

    int*  label_data_buffer		= (int*)  buffer_datas[1];
    float*  bbox_data_buffer	= (float*)buffer_datas[2];
    float*  score_data_buffe	= (float*)buffer_datas[3];
    int*  num_data_buffe		= (int*)  buffer_datas[4];

    int offset_label    = 0;
	int offset_bbox     = 0;
	int offset_score    = 0;
	int offset_num      = 0;

	Dims dim = engine->GetBindingDimensions(1);

    offset_label = volume(dim);

	dim = engine->GetBindingDimensions(2);

    offset_bbox = volume(dim);

	dim = engine->GetBindingDimensions(3);

    offset_score = volume(dim);

	dim = engine->GetBindingDimensions(4);

    offset_num = volume(dim);

    for(int b = 0; b < batch; b++) 
    {
        int valid_num = num_data_buffe[b*offset_num];
        if(valid_num > 9999)
        {
            valid_num = 0;
        }
        //printf("----valid_num:%d\n", valid_num);
        std::vector<DL_Detect_Result_ST> result;
        result.clear();

        float* pScore   = score_data_buffe  + b*offset_score;
		float* pBbox    = bbox_data_buffer  + b*offset_bbox;
		int *  pLabel   = label_data_buffer + b*offset_label;

        DL_Resize_Param_ST &stResizeParam = stResizeParamSet[b];

        for(int v = 0; v < valid_num; v++)
        {
            int label   = pLabel[2*v+1];
			float score = pScore[v];
			float min_x = pBbox[4*v];
			float min_y = pBbox[4*v+1];
			float max_x = pBbox[4*v+2];
			float max_y = pBbox[4*v+3];

            DL_Detect_Result_ST res;
            res.iLabel  = label;
            res.fScore  = score;
            res.x       = (min_x - stResizeParam.pad_x)/stResizeParam.fRatio;
            res.y       = (min_y - stResizeParam.pad_y)/stResizeParam.fRatio;
            res.width   = (max_x - min_x)/stResizeParam.fRatio;
            res.height  = (max_y - min_y)/stResizeParam.fRatio;

            result.push_back(res);
        }
        _Results.push_back(result);
    }

    return 0;
}

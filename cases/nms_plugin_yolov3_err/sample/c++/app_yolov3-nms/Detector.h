#ifndef _YOLOV5_NMS_DETECTOR_H_
#define _YOLOV5_NMS_DETECTOR_H_
#include <memory>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <fstream>
#include "dlnne.h"
#include "dlnne_build_modulator.h"
#include "dljpudecode.h"

#ifndef DL_OPENCV_ENABLE
#define DL_OPENCV_ENABLE
#endif
typedef  struct 
{
    int iLabel;
    float fScore;
    int x;
    int y;
    int width;
    int height;
} DL_Detect_Result_ST;

class DlDetector
{
public:
    DlDetector();
    ~DlDetector();
    // name for caffe model 
    // id for gpu
    void LoadModel(	const std::string& modelFile, int dlgpu_id, int maxBatch);

    int Detect(std::vector<DlImage*> Src, std::vector< std::vector<DL_Detect_Result_ST> > &_Results);
public:

	 
private:
    dl::nne::Engine* engine = nullptr;
    dl::nne::Network* network = nullptr;

    dl::nne::Builder* builder = nullptr;
    dl::nne::ExecutionContext* context = nullptr;   

    std::vector<void *>  bindings; 
    std::vector<size_t>  buffer_sizes;
    std::vector<void *>  buffer_datas;

    int m_iMaxBatch;

    // void *in_dev_buf;
    // int in_max_size;
    int m_dlGpuId;
    cudaStream_t m_Stream;
};
#endif// end _YOLOV5-NMS_DETECTOR_H_

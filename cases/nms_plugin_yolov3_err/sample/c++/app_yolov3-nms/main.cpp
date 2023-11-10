#include <dirent.h>
#include <cstring>
#include <vector>
#include "dljpudecode.h"
#include "util.h"
#include "Detector.h"
#ifdef DL_OPENCV_ENABLE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#endif

using namespace std;

void getFiles(std::string path, std::vector<std::string> &files)
{
  DIR *dir;
  struct dirent *ptr;

  if ((dir = opendir(path.c_str())) == NULL)
  {
    perror("Open dir error...");
    exit(1);
  }

  while ((ptr = readdir(dir)) != NULL)
  {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0){
    continue;
    }else if (ptr->d_type == 8){
    std::string filename = ptr->d_name;
    std::string exeName = filename.substr(filename.find_last_of('.'));
    if(exeName == ".jpg"){
      files.push_back(path + ptr->d_name);
    }
  }else if (ptr->d_type == 10){
      continue;
  }else if (ptr->d_type == 4)
    {
      //files.push_back(ptr->d_name);
      getFiles(path + ptr->d_name + "/", files);
    }
  }
  closedir(dir);
}

int main(int argc, char* argv[])
{
  if(argc < 4)
  {
    printf("argv is model_path images_path batch_size\n");
    return 0;
  }
  int dlgpu_id = 0;
  int batch = atoi(argv[3]);
  DlDetector detector;

  detector.LoadModel(argv[1], dlgpu_id, batch);

  std::cout << "head init sucess" << std::endl;
  vector<std::string> fn;
  getFiles(argv[2], fn);
  
  // int iNum = fn.size();
  // for(int i = 0; i < 64 - iNum; i++)
  // {
  //   fn.push_back(fn[0]);
  // }

  printf("-----fn.size: %d\n", fn.size());

  DL_JPEG_DEVICE  device;
  DL_JPEG_SESSION session;
  DLJPU_INIT_PARA jpuinit;
  DL_JPEG_DECODE_PARAMS paras = {0};
  // Select a JPU in this GPU device
  jpuinit.info.channelMask    = 0x01;
  jpuinit.info.clusterMask    = 0x01;
  jpuinit.deviceID            = dlgpu_id;
  device                      = dljpu_init(jpuinit);
  session                     = dljpu_getsession(device);

  vector<DlImage*> SrcList;

  SrcList.clear();
  for(int i = 0; i < fn.size(); i++)
  {
      DlImage* image = new DlImage();
      dljpu_decode(*image, device, session, paras, fn[i]);
      if (image->format != DL_JPEG_PIXEL_FORMAT_YUV420P) {
          cout << "not DL_JPEG_PIXEL_FORMAT_YUV420P" << endl;
          continue;
      }

      if(SrcList.size()< batch)
      {
          SrcList.push_back(image);
      }
      if(SrcList.size() < batch && i != fn.size()-1 )
      {
        continue;
      }
      //printf("%s\n",fn[i].c_str());  
      std::vector< std::vector<DL_Detect_Result_ST> > Results;

      util::Timer timer_pre;
      timer_pre.start();
      detector.Detect(SrcList, Results);

      timer_pre.stop();
      timer_pre.ShowLastTime("----process ");

      for(int j = 0;j<SrcList.size();j++)
      {
        std::vector<DL_Detect_Result_ST> rects = Results[j];
        printf("detection num %d\n", rects.size());
#ifdef DL_OPENCV_ENABLE
        int src_w = SrcList[j]->iWidth, src_h = SrcList[j]->iHeight;
        cv::Mat img_yuv = cv::Mat(src_h*3/2, src_w, CV_8UC1);

        cudaMemcpy((char*)(img_yuv.data), (char*)(SrcList[j]->data), src_w*src_h*3/2,cudaMemcpyDeviceToHost);
        cv::Mat gbr24;
        cvtColor(img_yuv,gbr24, cv::COLOR_YUV2BGR_I420);

        for(int k = 0; k < rects.size(); k++ ){
          printf("label:%d score:%f x:%d y:%d width:%d height:%d\n", rects[k].iLabel, rects[k].fScore, 
            (int)((rects[k].x)), (int)((rects[k].y)), (int)(rects[k].width), (int)(rects[k].height));

          cv::Rect segrect;

          segrect.x = (int)((rects[k].x));
          segrect.y = (int)((rects[k].y));

          segrect.width = (int)(rects[k].width);
          segrect.height = (int)(rects[k].height);

          int label = rects[k].iLabel;
          float score = rects[k].fScore;

          cv::rectangle(gbr24, segrect, cv::Scalar(255, 0, 255), 2);
          cv::Point pt;
          char text[256] = { 0 };
          pt.x = segrect.x;
          pt.y = segrect.y;
          sprintf(text, "type-%d, score-%.2f",label, (double)score);
          cv::putText(gbr24, text, pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 1);
        }

        char file_name[256];
        static int savenum = 0;
        savenum++;
        sprintf(file_name, "./%d_result.jpg", savenum);
        imwrite(file_name, gbr24);
#else 
        for(int k = 0; k < rects.size(); k++ ){
          printf("label:%d score:%f x:%d y:%d width:%d height:%d\n", rects[k].iLabel, rects[k].fScore, 
            (int)((rects[k].x)), (int)((rects[k].y)), (int)(rects[k].width), (int)(rects[k].height));
        }        
#endif
        delete SrcList[j];     
      }
      /**/  
      SrcList.clear(); 
  } 
}



#ifndef __DLJPU_DECODE_HPP__
#define __DLJPU_DECODE_HPP__

#include <string>
#include <iostream>
#include <stdio.h>
#include <cstdint>
#include "cuda_runtime.h"
#include <dljpeg.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <thread>
#include <vector>

typedef struct DLJPU_INIT_PARA{
  DL_JPEG_DEVICE_INFO   info;
  int   deviceID;
}DLJPU_INIT_PARA;

DL_JPEG_DEVICE dljpu_init(DLJPU_INIT_PARA initPara);

DL_JPEG_SESSION  dljpu_getsession(DL_JPEG_DEVICE device);

class DlImage
{
public:
  virtual ~DlImage()
  {
    if(data)
    {
      cudaFree(data);
      data = nullptr;
    }
  }
public:
  DL_JPEG_PIXEL_FORMAT format = DL_JPEG_PIXEL_FORMAT_YUV420P;
  int   iHeight = 0;
  int   iWidth = 0;
  int   iChannel = 0;
  int   iLength = 0;
  char  *data = nullptr;
  int   iFrameId = -1;
};

int dljpu_decode(DlImage &image,\
          DL_JPEG_DEVICE  device,\
          DL_JPEG_SESSION  session,\
          DL_JPEG_DECODE_PARAMS   paras,\
          std::string inputFilename);

void dljpv_session_del(DL_JPEG_SESSION  session);
#endif 
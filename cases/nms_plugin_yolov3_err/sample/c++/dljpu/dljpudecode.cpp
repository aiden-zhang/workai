
#include <string>
#include <iostream>
#include <stdio.h>
#include <cstdint>
#include "cuda_runtime.h"
#include <dljpeg.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include "dljpudecode.h"
using namespace std;
using namespace std::chrono;



#define DLJPEG_RESULT_CHECK(x)                                             \
  do {                                                                     \
    if ((x != DL_JPEG_RESULT_SUCCESS)) {                                   \
      printf("%s:%d %s error code %d\n", __FILE__, __LINE__, __func__, x); \
      goto EXIT;                                                           \
    }                                                                      \
  } while (0)

DL_JPEG_DEVICE dljpu_init(DLJPU_INIT_PARA initPara)
{
  DL_JPEG_RESULT result = DL_JPEG_RESULT_FAILED;
  DL_JPEG_DEVICE_INFO info = initPara.info;
  DL_JPEG_DEVICE device = nullptr;
  DL_JPEG_DEVICE_PROP prop;
  cudaSetDevice(initPara.deviceID);
  // Select a JPU in this GPU device
  result = dljpegGetDevice(&device, &info);

  DLJPEG_RESULT_CHECK(result);
  dljpegGetDeviceProp(device, &prop);
  DLJPEG_RESULT_CHECK(result);
  //cout << "jpu cluster " << prop.cluster << " channel " << prop.channel << endl;
EXIT:
  return device;
}

DL_JPEG_SESSION  dljpu_getsession(DL_JPEG_DEVICE device)
{
  DL_JPEG_RESULT result=DL_JPEG_RESULT_FAILED;
  DL_JPEG_DEVICE tdevice = (DL_JPEG_DEVICE)device;
  DL_JPEG_SESSION session = nullptr;
// Create a session for decode
  result = dljpegCreateSession(&session, tdevice);
  DLJPEG_RESULT_CHECK(result);
EXIT:
 return session;
}


uint32_t dljpeg_getDataChannel(DL_JPEG_PIXEL_FORMAT format)
{
  uint32_t channel = 0;
  switch(format)
  {
    case DL_JPEG_PIXEL_FORMAT_YUV420P:
    case DL_JPEG_PIXEL_FORMAT_YUV422P:
    case DL_JPEG_PIXEL_FORMAT_YUV440P:
    case DL_JPEG_PIXEL_FORMAT_YUV444P:
    case DL_JPEG_PIXEL_FORMAT_YUV400P:
    { 
      channel = 3;
      break;
    }
    case  DL_JPEG_PIXEL_FORMAT_NV12:
    case  DL_JPEG_PIXEL_FORMAT_NV21:
    case  DL_JPEG_PIXEL_FORMAT_NV16:
    case  DL_JPEG_PIXEL_FORMAT_NV61:
    case  DL_JPEG_PIXEL_FORMAT_NV24:
    case  DL_JPEG_PIXEL_FORMAT_NV42:
    {
      channel = 2;
      break;
    }
    case  DL_JPEG_PIXEL_FORMAT_YUYV422:
    case  DL_JPEG_PIXEL_FORMAT_UYVY422:
    case  DL_JPEG_PIXEL_FORMAT_YVYU422:
    case  DL_JPEG_PIXEL_FORMAT_VYUY422:
    case  DL_JPEG_PIXEL_FORMAT_YUV444:
    {
      channel = 1;
      break;
    }
    default:
    {
      cout << "error:invialid format of decode \n" << format << endl;
      break;
    }
  }
 return channel;
}

int dljpu_decode(DlImage &image,\
          DL_JPEG_DEVICE  device,\
          DL_JPEG_SESSION  session,\
          DL_JPEG_DECODE_PARAMS   paras,\
          string inputFilename)
{
  DL_JPEG_RESULT result;
  time_point<high_resolution_clock> startTimePoint;
  time_point<high_resolution_clock> lastTimePoint;
  void *inputBuffer=nullptr;
  void *outputBuffer[3];
  size_t outputBufferSize[3];
  uint32_t outputBufferStride[3] = {0};
  void *tempBuffer = nullptr;
  int imglength = 0;
  void *jpegData = nullptr;
  uint32_t jpegDataSize = 0;
  uint32_t sosOffset;
  uint32_t channel = 0;
  int iret =-1;
  DL_JPEG_HEADER_INFO *headerInfo = nullptr;
  DL_JPEG_PIXEL_FORMAT format;
  DL_JPEG_DECODE_PARAMS decodeParams = paras;

  ifstream inputFile;
  // Open input output file
  inputFile.open(inputFilename.c_str(), std::ios::in | std::ios::binary);
  //printf("%s\n",inputFilename.c_str());
  if (!inputFile.is_open()) {
    cout << "Unable to open input file: " << inputFilename << endl;
    cout << "test::dljpegDecodeSample FAILED\n";
    goto EXIT;
  }
  inputFile.seekg(0, std::ios_base::end);
  jpegDataSize = inputFile.tellg();
  inputFile.seekg(0, std::ios_base::beg);

  //cout << "input jpeg file size " << jpegDataSize << endl;
  // Alloc host mem
  if (cudaMallocHost(&jpegData, jpegDataSize) != cudaSuccess) {
    cout << "cudaMallocHost failed\n";
    cout << "test::dljpegDecodeSample FAILED\n";
    goto EXIT;
  }
  // Read jpeg data from input file
  jpegDataSize = inputFile.read(reinterpret_cast<char *>(jpegData), jpegDataSize).gcount();

  // Parse jpeg header
  result = dljpegParseHeader(&headerInfo, &sosOffset, jpegData, jpegDataSize);
  DLJPEG_RESULT_CHECK(result);

  if (headerInfo->compInfo[0].hSampFactor == 1 &&
      headerInfo->compInfo[0].vSampFactor == 1) {
    if (headerInfo->numComponents == 1)
      format = DL_JPEG_PIXEL_FORMAT_YUV400P;
    else
      format = DL_JPEG_PIXEL_FORMAT_YUV444P;
  } else if (headerInfo->compInfo[0].hSampFactor == 2 &&
             headerInfo->compInfo[0].vSampFactor == 1) {
    format = DL_JPEG_PIXEL_FORMAT_YUV422P;
  } else {
    format = DL_JPEG_PIXEL_FORMAT_YUV420P;
  }

  channel = dljpeg_getDataChannel(format);
  // Get output buffer size.
  // Output yuv width and height is round up to multiple of 16.
  decodeParams.headerInfo = headerInfo;
  result = dljpegGetImageBufferSize(outputBufferSize, &decodeParams, format);
  DLJPEG_RESULT_CHECK(result);
  // Allocate input buffer
  result = dljpegMalloc(&inputBuffer, device, jpegDataSize - sosOffset);
  DLJPEG_RESULT_CHECK(result);
  // Copy jpeg data to input buffer
  cudaMemcpy(inputBuffer, reinterpret_cast<uint8_t *>(jpegData) + sosOffset,
               jpegDataSize - sosOffset, cudaMemcpyHostToDevice);
  
  if(format == DL_JPEG_PIXEL_FORMAT_YUV400P)
  {
    outputBufferSize[1] = outputBufferSize[0];
    outputBufferSize[2] = outputBufferSize[0];
  }
  
  for (int i = 0; i < channel; i++) {
      result = dljpegMalloc(&outputBuffer[i], device, outputBufferSize[i]);
      imglength += outputBufferSize[i];
      if(result == -6)
      {
          cout << "chanl " << channel << " formant " << format << endl;
          cout << "w " << headerInfo->width << " h " << headerInfo->height << " len "<< imglength <<" bufsize " << outputBufferSize[i] << endl;
         for(int q =0;q< channel ;q++)
         {
           cout << "obuf " << outputBuffer[q] << " device " << device << " obufSize " << outputBufferSize[q] << endl;
         }
      }
      DLJPEG_RESULT_CHECK(result);
    }
 // Decode
  result = dljpegDecode(session, &decodeParams, inputBuffer,
                            jpegDataSize - sosOffset, outputBuffer,
                            outputBufferStride, format);
  DLJPEG_RESULT_CHECK(result);
  
  #if 1
  if (cudaMalloc(&image.data, imglength) == cudaSuccess)
  {
    char* tempBuffer = image.data;
    image.iLength = imglength;
    image.iWidth = (headerInfo->width + 15) / 16 * 16;
    image.iHeight = (headerInfo->height + 15) / 16 * 16;
    image.format = format;
    image.iChannel = channel;
    for(int i=0; i<channel; i++)
    {
      //printf("jpegdecode %zu \n", outputBufferSize[i]);
      cudaMemcpy(tempBuffer, outputBuffer[i], outputBufferSize[i], cudaMemcpyDeviceToDevice);
      tempBuffer+=outputBufferSize[i];
    }
  }
  #endif


 RET:
    iret =0;
 EXIT:
  if (inputBuffer) dljpegFree(inputBuffer);
  for (int i = 0; i < channel; i++) {
      if (outputBuffer[i]) dljpegFree(outputBuffer[i]);
  }

  if (jpegData) 
  {
    cudaFreeHost(jpegData);
  }
  if (headerInfo)
  {
    dljpegFreeHeader(headerInfo);
  }
  inputFile.close();
 
  return iret;
}



void dljpv_session_del(DL_JPEG_SESSION  session)
{

   if (session) dljpegDestroySession(session);
}
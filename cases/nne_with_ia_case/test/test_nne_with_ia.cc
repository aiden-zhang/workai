#include <stdio.h>
#include "cuda_fp16.h"
#include "image_proc.h"
#include "util.h"
#include "case_register.h"
#include <dlnne.h>
#include <cuda_runtime_api.h>
#include <cu_ext.h>

using namespace util;

extern int nne_run(void* inputFrame, int inputSize, cudaStream_t cudaStream);

std::string RunOpkernelAndNne(cudaStream_t stream, const int call_times,
                         uint8_t* in_cpu, uint8_t* out_cpu,
                         int w_in, int h_in,
                         int w_out, int h_out, int mode) {
  std::string str;
  if (stream == NULL) {
    str = "default_stream_";
  } else {
    str = "";
  }
  std::string calc_mode = "";
  if (mode == NEAREST)
    calc_mode = "nearest";
  if (mode == BILINEAR)
    calc_mode = "bilinear";
  std::string case_name = "";
  printf("========\"call_times \" kernel launch and nne infer ========\n");
  case_name = "gray_" + calc_mode + "_resize_and_nne_infer_" + str + std::to_string(w_in) + "_" + std::to_string(h_in) + "_" +
        std::to_string(w_out) + "_" + std::to_string(h_out) + "_" + std::to_string(call_times);
  std::cout << "\n<customer_bugs_test_case::"<< case_name<<" RUN>\n" << std::endl;

  uint8_t *in_dev_rgb;
  int in_size = 3 * w_in * h_in;     //1920*1080
  CUDA_CHECK(cudaMalloc(&in_dev_rgb, sizeof(uint8_t) * in_size));
  CUDA_CHECK(cudaMemcpyAsync(in_dev_rgb, in_cpu, sizeof(uint8_t) * in_size, cudaMemcpyHostToDevice, stream));

  // CUDA_CHECK(cudaStreamSynchronize(stream));  
  uint8_t *bufferTmpRgb; //3*1280*720
  CUDA_CHECK(cudaMalloc(&bufferTmpRgb, sizeof(uint8_t) *1280*720*3));
  CUDA_CHECK(cudaMemset(bufferTmpRgb, 0, sizeof(uint8_t) *1280*720*3));

  uint8_t * out_tmp; //1280*720
  CUDA_CHECK(cudaMalloc(&out_tmp, sizeof(uint8_t) * 1280*720));
  CUDA_CHECK(cudaMemset(out_tmp, 0, sizeof(uint8_t) * 1280*720));

  uint8_t *out_dev;
  int out_size =  w_out * h_out; //1000*1000
  CUDA_CHECK(cudaMalloc(&out_dev, sizeof(uint8_t) * out_size));
  CUDA_CHECK(cudaMemset(out_dev, 0, sizeof(uint8_t) * out_size));

  uint8_t *nne_input_rgb; //3*1000*1000
  int nne_input_size =  3 * w_out * h_out; //3*1000*1000
  CUDA_CHECK(cudaMalloc(&nne_input_rgb, sizeof(uint8_t) * nne_input_size));
  CUDA_CHECK(cudaMemset(nne_input_rgb, 0, sizeof(uint8_t) * nne_input_size));

  for(int i = 0; i < call_times; i++) {

    // cudaSetClusterMask(1 << 0);
    for(int i = 0; i < 3; i++) {
      GrayResizeNearest(in_dev_rgb + i*w_in*h_in, out_tmp, w_in, h_in, 1280, 720, stream);
      cudaMemcpy((char*)bufferTmpRgb+i*1280*720, out_tmp, 1280*720, cudaMemcpyDeviceToDevice);
    }

    for(int i = 0; i < 3; i++) {
      GrayResizeNearest(bufferTmpRgb + i*1280*720, out_dev, 1280, 720, 1000, 1000, stream);
      cudaMemcpy((char*)nne_input_rgb+i*out_size, out_dev, out_size, cudaMemcpyDeviceToDevice);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    nne_run(nne_input_rgb, nne_input_size, stream);
    // CUDA_CHECK(cudaMemcpyAsync(out_cpu, out_dev[0], sizeof(uint8_t) * out_size, cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));
}
 
  CUDA_CHECK(cudaFree(in_dev_rgb));
  CUDA_CHECK(cudaFree(bufferTmpRgb));
  CUDA_CHECK(cudaFree(out_tmp));
  CUDA_CHECK(cudaFree(out_dev));
  CUDA_CHECK(cudaFree(nne_input_rgb));
  return case_name;
}

void TestRun(int w_in, int h_in, int w_out, int h_out, const int call_times, int mode) {

  int in_size  =  w_in*h_in*3;
  int out_size =  w_out*h_out*3;
  uint8_t *in_cpu = NULL;
  uint8_t *out_cpu = NULL;
  cudaMallocHost(&in_cpu, in_size * sizeof(uint8_t));
  cudaMallocHost(&out_cpu, out_size * sizeof(uint8_t));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::string calc_mode = "";
  if (mode == NEAREST)
      calc_mode = "nearest";
  if (mode == BILINEAR)
      calc_mode = "bilinear";

  char in_file[100] = {""};
  sprintf(in_file, "../res/%dx%d.rgb", w_in, h_in);
  FILE* fp;
  fp = fopen(in_file, "rb");
  if (NULL != fp) {
      fread(in_cpu, 1, in_size, fp);
      fclose(fp);
  }

  std::string case_name = RunOpkernelAndNne(stream, call_times, in_cpu, out_cpu, w_in, h_in, w_out, h_out, mode);
  cudaFreeHost(in_cpu);
  cudaFreeHost(out_cpu);
  cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
  // if (argc < 6) {
  //   printf("Usage: \n"
  //          "  ./test_resize [input_width] [input_height] [output_width] [output_height] [call_times] [stream]");
  //   return -1;
  // }
  int w_in = 1920; //atoi(argv[1]);
  int h_in = 1080; //atoi(argv[2]);
  int w_out = 1000;//atoi(argv[3]);
  int h_out = 1000;//atoi(argv[4]);
  const int call_times = 1;//atoi(argv[5]);
  
  printf("input_width: %d, input_height: %d, output_width: %d, output_height: %d, call_times: %d\n",
    w_in, h_in, w_out, h_out, call_times);
  
  int mode = NEAREST;
  TestRun(w_in, h_in, w_out, h_out, call_times, mode);

  return 0;
}





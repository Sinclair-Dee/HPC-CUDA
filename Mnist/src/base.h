/********************************************
//base.h
********************************************/

#ifndef _BASE_H_
#define _BASE_H_

//IO
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

//thrust for CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

//System and Algorithm
#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>

//String
#include <cstring>
#include <string>

//常量定义区
#define TILE_WIDTH 16
#define MINIBATCH 1000
#define LEARNIG_RATE 2e-4
#define LAMDA (3e-1)
#define CONV_KERNEL_SIZE 5 // max conv kernel size
#define LABEL_ONE 1
#define LABEL_ZERO 0
#define NUM_TRAIN 60000
#define NUM_TEST  1000 // max conv kernel size0
#define RAW_DIM 28
#define RAW_PIXELS_PER_IMG 784
#define RAW_DIM_PADDING 32
#define RAW_PIXEL_PER_IMG_PADDING 1024
#define MNIST_SCALE_FACTOR 0.00390625
#define MAXBYTE 255
#define gpuError(ans) { gpuAssert((ans),__FILE__, __LINE__);}
inline void gpuAssert(cudaError_t Err, sonst char *file, int *line,bool abort = true){
  if(Err != cudaSuccess){
    fprintf(stderr,"GPUassert:%s %s %d\n",cudaGetErrorString(Err),file,line);
    if(abort) exit(code);
  }
}

namespace GPU_Scope{

void ClearScreen();
void printMNIST_H_W_row_col_for_Main(thrust::device_vector<float> &DATA_, int Height, int Width, int row_start_index, int col_start_index,
                                     int row_num, int col_num, int row_interval, int col_interval, char* str);

/***** Function declarations *****************************/
void printMNIST(thrust::host_vector<float>& data);
void read_data(const char* datapath, thrust::host_vector<thrust::host_vector<float> &data>);
void read_data_no_padding(const char* datapath, thrust::host_vector<thrust::host_vector<float> &data>);
void read_label(const char* datapath, thrust::host_vector<thrust::host_vector<float> &data>);
void ConvLayerForward(int batch, float *FM_in, int CH_in, int H_in, int W_in
                      float *W, int W_h_w, float *FM_out, int CH_out);
/***** Function declarations end *************************/

/***** CUDA Kernel declarations *****************************/

__global__
void ConvLayerForwardGPUnaive(float *FM_in, float *W, float *FM_out,
          int CH_in, int H_in, int W_in, int W_out, int W_h_w, int CH_out);
__global__
void ConvLayerForwardGPUtiled(float *FM_in, float  *W, float *FM_out,
	        int CH_in, int H_in, int W_in, int W_out, int W_h_w, int CH_out);
__global__
void ConvLayerBackwardGPUnaive(float *FM_in, float *W, float *FM_out,
	        int CH_in, int H_in, int W_in, int W_out, int W_h_w, int CH_out);
__global__
void unroll_Kernel(int CH_in, int H_in, int W_in, int W_h_w, float *FM_in, float *Unroll_FM_in);
__global__
void col2im_Kernel(int C, int H_in, int W_in, int K, float* FM_in, float* FM_Unroll);
__global__
void GEMM(float *MatrixM, float* MatrixN, float* MatrixOut,
          int M_height_in, int M_width_N_height_in, int N_width_in,
          int height_out, int width_out);
__global__
void GEMMwithBias(float *MatrixM, float* MatrixN, float* MatrixOut, float* B,
          int M_height_in, int M_width_N_height_in, int N_width_in,
          int height_out, int width_out);
/***** CUDA Kernel declarations end ******************************/

class Convolution {//M*C*H*W
public:
  void init(int minib, int Inputimage_h, int Inputimage_w, int Inputimage_ch, int W_w_h, int W_ch);
  void forward_CPU();
  void forward_GPU_naive();
  void forward_GPU_tiled();
  void forward_GPU_gemm();
  void forward_cpu_test(thrust::host_vector &input, int test_number);
  void forward_gpu_test(thrust::device_vector &input, int test_number);
  void backward_GPU_native();
  void backward_gpu_gemm();
  void backward_col2im_gpu_test(int test_number);
  void backward();
  void ConvLayerBackwardXgrad(int N, int M, int C, int H_in,int W_in,
                              int K, float *dE_dy, float *W, float *dE_dx);
  void ConvLayerBackwardWgrad(int N, int M, int C, int H_in, int W_in
                              int K, float *dE_dy, float *FM_in float *dE_dw);

  thrust::host_vector<float> host_FM_in;
  thrust::host_vector<float> host_W;
  thrust::host_vector<float> host_Bias;
  thrust::host_vector<float> host_Wgrad;
  thrust::host_vector<float> host_bgrad;
  thrust::host_vector<float> host_FM_out;
  thrust::host_vector<float> host_Unroll_FM_in;

  thrust::device_vector<float> device_FM_in;
  thrust::device_vector<float> device_W;
  thrust::device_vector<float> device_WT;
  thrust::device_vector<float> device_Bias;
  thrust::device_vector<float> device_Wgrad;
  thrust::device_vector<float> device_Wgrad_Temp;
  thrust::device_vector<float> device_bgrad;
  thrust::device_vector<float> device_FM_out;
  thrust::device_vector<float> device_Unroll_FM_in;
  thrust::device_vector<float> device_Unroll_FM_inT;

  int MiniBatch;
  int W_width;
  int W_height;
  int W_width_height;
  int W_channel;
  int FM_in_width;
  int FM_in_height;
  int Unroll_FM_in_width;
  int Unroll_FM_in_height;
  int Inputimage_height;
  int Inputimage_width;
  int Inputimage_channel;
  int FM_out_width;
  int Fm_out_height;
  int Outputimage_width;
  int Outputimage_height;
  int Outputimage_channel;

}

class FullyConnect {
public:
  void init( int minib, int Inputimage_w_W_h, int W_w);
  void forward(thrust::device_vector<float> &input);
  void foreard();
  void forward_gpu_test(thrust::device_vector<float> &input, int test_number);
  void backward();

  //M*(C*H*W)
  thrust::host_vector<float> host_FM_in;
  thrust::host_vector<float> host_W;
  thrust::host_vector<float> host_Bias;
  thrust::host_vector<float> host_Wgrad;
  thrust::host_vector<float> host_bgrad;
  thrust::host_vector<float> host_FM_out;

  thrust::device_vector<float> device_FM_in;
  thrust::device_vector<float> device_FM_inT;
  thrust::device_vector<float> device_W;
  thrust::device_vector<float> device_WT;
  thrust::device_vector<float> device_Bias;
  thrust::device_vector<float> device_Wgrad;
  thrust::device_vector<float> device_Bgrad;
  thrust::device_vector<float> device_FM_out;
  thrust::device_vector<float> device_FM_outT;

  int FM_in_width;
  int FM_in_height;
  int FM_inT_width;
  int FM_inT_height;
  int W_width;
  int W_height;
  int W_width_height;
  int WT_width;
  int WT_height;
  int WT_width_height;
  int FM_out_width;
  int Fm_out_height;
  int FM_outT_width;
  int Fm_outT_height;

}

class Pool {//// M*C*H*W
public:
  void init();
  void polligLayer_forward();





  int MiniBatch;
  int FM_in_width;
  int FM_in_height;
  

  int pool_size;
  int W_width;
  int W_height;
  int W_width_height;
  int W_channel;

  int Unroll_FM_in_width;
  int Unroll_FM_in_height;
  int Inputimage_height;
  int Inputimage_width;
  int Inputimage_channel;
  int FM_out_width;
  int Fm_out_height;
  int Outputimage_width;
  int Outputimage_height;
  int Outputimage_channel;





}

class Softmax {

}

}


#endif

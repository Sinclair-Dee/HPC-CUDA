/********************************************
//Net.h
**********************i**********************/

#ifndef _NET_H_
#define _NET_H_

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
#define TILE_WIDTH 32
#define MINIBATCH 1000
#define LEARNIG_RATE 2e-4
#define LAMDA (3e-1)
#define CONV_KERNEL_SIZE 13
#define LABEL_ONE 1
#define LABEL_ZERO 0
#define NUM_TRAIN 60000
#define NUM_TEST  10000
#define RAW_DIM 28
#define RAW_PIXELS_PER_IMG 784
#define RAW_DIM_PADDING 32
#define RAW_PIXEL_PER_IMG_PADDING 1024
#define MNIST_SCALE_FACTOR 0.00390625
#define MAXBYTE 255

#define gpuError(ans) { gpuAssert((ans),__FILE__, __LINE__);}
inline void gpuAssert(cudaError_t ans, const char *file, int line, bool abort = true){
  if(ans != cudaSuccess){
    fprintf(stderr,"GPU_Assert:%s%s%d\n", cudaGetErrorString(ans), file, line);
    if(abort) exit(ans);
  }
}

namespace GPU_Scope{

void ClearScreen();
void printMNIST_H_W_row_col_for_Main(thrust::device_vector<float> &DATA_, int Height, int Width, int row_start_index, int col_start_index, int row_num, int col_num, int row_interval, int col_interval, char* str);

/***** Function declarations *****************************/
void printMNIST(thrust::host_vector<float>& data);
void read_data(const char* datapath, thrust::host_vector<thrust::host_vector<float> &data>);
void read_data_no_padding(const char* datapath, thrust::host_vector<thrust::host_vector<float> &data>);
void read_label(const char* datapath, thrust::host_vector<thrust::host_vector<float> &data>);

/***** Function declarations end *************************/

  
/***** CLASS declarations ******************************/
class FullyConnect {

}

class Convolution {//M*C*H*W
public:
  void init(int minib, int X_h, int X_w, int X_ch, int W_w_h, int W_ch);
   
  thrust::host_vector<float> host_FM_in;
  thrust::host_vector<float> host_W
  thrust::host_vector<float> b_data
  thrust::host_vector<float> Wgrad_data
  thrust::host_vector<float> O
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
  thrust::host_vector<float>
thrust::host_vector
thrust::host_vector

  
  int Mini_Batch;
  int W_width;
  int W_height;
  int W_width_height;
  int W_channel;
  int Inputimage_height;
  int Inputimage_channel;
  int Inputimahe_channel;
  int FM_in_width;
  int FM_in__height;
  int Outputimage_width;
  int Outputimage_height; 
  int Outputimage_channel;
  int FM_out__width;
  int Fm_out__height;
  int Unroll_X_width;
  int Unroll_X_height;


  






}

class Pool {

}

class Softmax {

}

class Net{

}

class Net_CPU: public Net{

}

class Net_GPU_Naive: public Net{

}

class Net_GPU_Tiled: public Net{

}

class Net_GPU_Gemm: public Net{

}

class Net_GPU_test: public Net{

}

inline void gpuAssert(cudaError_t Err, sonst char *file, int *line,bool abort = true){
  if(Err != cudaSuccess){
    fprintf(stderr,"GPUassert:%s %s %d\n",cudaGetErrorString(Err),file,line);
    if(abort) exit(code);
  }
}






}
#endif

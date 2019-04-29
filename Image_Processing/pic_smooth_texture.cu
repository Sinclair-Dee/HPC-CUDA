#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "stdio.h"

//声明纹理内存
texture<uchar4,cudaTextureType2D,cudaReadModeNormalizedFloat> tex;

//smooth kernel
__global__
void smooth_kernel(char *img, int width, int height, int channels){
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int offset = x + y * blockDim.x +
}

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
  unsigned int offset = x + y * blockDim.x * gridDim.x;

  //归一化
  float u = x/(float)width;
  float v = y/(float)height;

  //获取纹理内存中的输入图像信息
  float4 pixel    = tex2D(tex,x,y);
  float4 left     = tex2D(tex,x-1,y);
  float4 right    = tex2D(tex,x+1,y);
  float4 top      = tex2D(tex,x,y-1);
  float4 botton   = tex2D(tex,x,y+1);

  //获取输出img
  img[(y*width+x)*channels+0] = (left.x+right.x+top.x+botton.x)/4*255;
  img[(y*width+x)*channels+1] = (left.y+right.y+top.y+botton.y)/4*255;
  img[(y*width+x)*channels+2] = (left.z+right.z+top.z+botton.z)/4*255;
  img[(y*width+x)*channels+3] = 0;
}

int main(){
  cv::Mat src = imread("lena.jpg",IMREAD_COLOR);
  cv::resize(src, src, Size(256, 256));
  cvtColor(src, src, CV_BGR2BGRA);
  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();
  int width = cols;
  int 
}

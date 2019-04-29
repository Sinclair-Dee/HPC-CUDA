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
  //unsigned int offset = x + y * blockDim.x * gridDim.x;

  //归一化时候的情况
  //float u = x/(float)width;
  //float v = y/(float)height;

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
  cv::Mat src = imread("lena.jpg",cv::IMREAD_COLOR);
  cv::resize(src, src, cv::Size(256, 256));
  cvtColor(src, src, CV_BGR2BGRA);
  int width = src.rows;
  int height = src.cols;
  int channels = src.channels();
  int imgsize = width * height * channels;

  //定义cuda数组
  cudaArray *cuArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  cudaMemcpyToArray(cuArray,0,0,src.data, imgsize, cudaMemcpyHostToDevice);

  //运行时纹理参照系属性
  tex.addressMode[0] = cudaAddressModeWrap;//循环寻址方式
  tex.addressMode[1] = cudaAddressModeWrap;
  tex.filterMode = cudaFilterModeLinear; //线性插值
  tex.normalized = false;

  //绑定
  cudaBindTextureToArray(tex,cuArray,channelDesc);

  //output of host and device  
  cv::Mat out = cv::Mat::zeros(width, height, CV_8UC4);
  char *dev_out = NULL;
  cudaMalloc((char**)&dev_out,imgsize);

  //invoke the kernel
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); 
  smooth_kernel<<<grid, block,0>>>(dev_out,width,height,channels);
  
  //copy result from device to host
  cudaMemcpy(out.data, dev_out, imgsize, cudaMemcpyDeviceToHost);
  
  //cv::imshow("origin");
  //cv::imshow("smooth_image",out);
  //waitKey(0);

  cv::imwrite("lena_smooth.jpg",out);

  cudaUnbindTexture(tex);
  cudaFree(dev_out);
  cudaFree(cuArray);

  return 0;  
}

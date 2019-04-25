#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace cv;

//声明CUDA纹理:texture<type,dimension,readtype> texreference;
texture <uchar4,cudaTextureType2D,cudaReadModeNormaliedFloat> refTex1;
texture <uchar4,cudaTextureType2D,cudaReadModeNormaliedFloat> refTex2;

////声明CUDA数组
cudaArray* cuArray1;
cudaArray* cuArray2;

//通道数
cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();

__global__ 
void weightAddKernel(uchar *pDstImgData, int imgHeight, int imgWidth,int channels){
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int tidy = blockDim.y * blockIdx.x + threadIdx.y;

  if(tidx<imgWidth && tidy<imgHeight){
    float4 lenaBGR, moonBGR;
    //使用tex2D函数采样纹理
    lenaBGR = tex2D(refTex1,tidx,tidy);
    moonBGR = tex2D(refTex2,tidx,tidy);

    int idx = (tidy * imgWidth + tidx) * channels;
    float alpha = 0.5;
    pDstImgData[idx+0]=(alpha*lenaBGR.x+(1-alpha)*moonBGR.x)*255;
    pDstImgData[idx+1]=(alpha*lenaBGR.y+(1-alpha)*moonBGR.y)*255;
    pDstImgData[idx+2]=(alpha*lenaBGR.z+(1-alpha)*moonBGR.z)*255;
    pDstImgData[idx+3]=0;
  }
}

int main(){
  cv::Mat lena=imread("lena.jpg");
  cv::Mat moon=imread("moon.jpg");

  cvtColor(lena,lena,CV_BGR2BGRA);
  cvtColor(moon,moon,CV_BGR2BGRA);
  int imgWidth = lena.cols;
  int imgHeight = lena.rows;
  int channels = lena.channels();

  //设置纹理属性
  cudaError_t t;
  refTex1.addressMode[0] = cudaAddressModeClamp;
  refTex1.addressMode[1] = cudaAddressModeClamp;
  refTex1.normallized = false;
  refTex1.filterMode = cudaFilterModeLinear;
  //绑定cuArray到纹理
  cudaMallocArray(&cuArray1,&cuDesc,imgWidth,imgHeight);
  t = cudaBindTextureToArray(refTex1,cudaArray1);
 
  refTex2.addressMode[0] = cudaAddressModeClamp;
  refTex2.addressMode[1] = cudaAddressModeClamp;

}

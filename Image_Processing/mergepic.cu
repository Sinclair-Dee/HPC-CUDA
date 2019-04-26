#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace cv;

//声明CUDA纹理:texture<type,dimension,readtype> texreference;
texture <uchar4,cudaTextureType2D,cudaReadModeNormalizedFloat> refTex1;
texture <uchar4,cudaTextureType2D,cudaReadModeNormalizedFloat> refTex2;

////声明CUDA数组
cudaArray* cuArray1;
cudaArray* cuArray2;

//通道数
cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();

__global__ 
void weightAddKernel(uchar *pDstImgData, int imgHeight, int imgWidth,int channels){
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

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

  //设置refTex1纹理属性
  cudaError_t t;
  refTex1.addressMode[0] = cudaAddressModeClamp;
  refTex1.addressMode[1] = cudaAddressModeClamp;
  refTex1.normalized = false;
  refTex1.filterMode = cudaFilterModeLinear;
  //绑定cuArray到纹理refTex1
  cudaMallocArray(&cuArray1,&cuDesc,imgWidth,imgHeight);
  t = cudaBindTextureToArray(refTex1,cuArray1);
  //设置refTex2纹理属性
  refTex2.addressMode[0] = cudaAddressModeClamp;
  refTex2.addressMode[1] = cudaAddressModeClamp;
  refTex2.normalized = false;
  refTex2.filterMode = cudaFilterModeLinear;
  //绑定cuArray到纹理refTex2
  cudaMallocArray(&cuArray2, &cuDesc, imgWidth, imgHeight);
  t = cudaBindTextureToArray(refTex2,cuArray2);
  
  //拷贝数据到cudaArray
  t = cudaMemcpyToArray(cuArray1,0,0,lena.data,imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyHostToDevice);
  t = cudaMemcpyToArray(cuArray2,0,0,moon.data,imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyHostToDevice);
  
  //输出图像
  Mat dstImg = Mat::zeros(imgHeight, imgWidth, CV_8UC4);
  uchar *pDstImgData = NULL;
  t = cudaMalloc((uchar **)&pDstImgData, imgHeight*imgWidth*sizeof(uchar)*channels);

  //invoke the kernel
  dim3 block(16,16);
  dim3 grid((imgWidth+block.x-1)/block.x, (imgHeight+block.y-1)/block.y);
  weightAddKernel<<<grid, block, 0>>>(pDstImgData, imgHeight, imgWidth, channels);
  t = cudaThreadSynchronize();

  //从GPU拷贝输出到CPU
  t=cudaMemcpy(dstImg.data, pDstImgData, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyDeviceToHost);

  //显示
//  namedWindow("show");
//  imshow("show",dstImg);
//  waitKey();
  
  //存储
  cv::imwrite("hill_merge.jpg",dstImg);
    
  //unbind and free memory
  cudaUnbindTexture(refTex1);
  cudaUnbindTexture(refTex2);
  cudaFreeArray(cuArray1);
  cudaFreeArray(cuArray2);

  return 0;
}

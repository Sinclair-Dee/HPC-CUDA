#include <opencv2/opencv.hpp> 
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace cv;

//声明CUDA纹理
texture <uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex1;
texture <uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex2;
//声明CUDA数组
cudaArray* cuArray1;
cudaArray* cuArray2;
//通道数
cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();


__global__ void weightAddKerkel(uchar *pDstImgData, int imgHeight, int imgWidth,int channels)
{
    const int tidx=blockDim.x*blockIdx.x+threadIdx.x;
    const int tidy=blockDim.y*blockIdx.y+threadIdx.y;

    if (tidx<imgWidth && tidy<imgHeight)
    {
        float4 lenaBGR,moonBGR;
        //使用tex2D函数采样纹理
        lenaBGR=tex2D(refTex1, tidx, tidy);
        moonBGR=tex2D(refTex2, tidx, tidy);

        int idx=(tidy*imgWidth+tidx)*channels;
        float alpha=0.5;
        pDstImgData[idx+0]=(alpha*lenaBGR.x+(1-alpha)*moonBGR.x)*255;
        pDstImgData[idx+1]=(alpha*lenaBGR.y+(1-alpha)*moonBGR.y)*255;
        pDstImgData[idx+2]=(alpha*lenaBGR.z+(1-alpha)*moonBGR.z)*255;
        pDstImgData[idx+3]=0;
    }
}

int  main()
{
    Mat Lena=imread("lena.jpg");
    Mat moon=imread("moon.jpg");
    cvtColor(Lena, Lena, CV_BGR2BGRA);
    cvtColor(moon, moon, CV_BGR2BGRA);
    int imgWidth=Lena.cols;
    int imgHeight=Lena.rows;
    int channels=Lena.channels();

    //设置纹理属性
    cudaError_t t;
    refTex1.addressMode[0] = cudaAddressModeClamp;
    refTex1.addressMode[1] = cudaAddressModeClamp;
    refTex1.normalized = false;
    refTex1.filterMode = cudaFilterModeLinear;
    //绑定cuArray到纹理
    cudaMallocArray(&cuArray1, &cuDesc, imgWidth, imgHeight);
    t = cudaBindTextureToArray(refTex1, cuArray1);

    refTex2.addressMode[0] = cudaAddressModeClamp;
    refTex2.addressMode[1] = cudaAddressModeClamp;
    refTex2.normalized = false;
    refTex2.filterMode = cudaFilterModeLinear;
     cudaMallocArray(&cuArray2, &cuDesc, imgWidth, imgHeight);
    t = cudaBindTextureToArray(refTex2, cuArray2);

    //拷贝数据到cudaArray
    t=cudaMemcpyToArray(cuArray1, 0,0, Lena.data, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyHostToDevice);
    t=cudaMemcpyToArray(cuArray2, 0,0, moon.data, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyHostToDevice);

    //输出图像
    Mat dstImg=Mat::zeros(imgHeight, imgWidth, CV_8UC4);
    uchar *pDstImgData=NULL;
    t=cudaMalloc(&pDstImgData, imgHeight*imgWidth*sizeof(uchar)*channels);

    //核函数，实现两幅图像加权和
    dim3 block(8,8);
    dim3 grid( (imgWidth+block.x-1)/block.x, (imgHeight+block.y-1)/block.y );
    weightAddKerkel<<<grid, block, 0>>>(pDstImgData, imgHeight, imgWidth, channels);
    cudaThreadSynchronize();

    //从GPU拷贝输出数据到CPU
    t=cudaMemcpy(dstImg.data, pDstImgData, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyDeviceToHost);

    //显示
    //namedWindow("show");
    //imshow("show", dstImg);
    //waitKey(0);i
  cv::imwrite("output.jpg",dstImg);
  return 0;
} 

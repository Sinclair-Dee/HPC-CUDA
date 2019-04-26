#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include<sys/time.h>
//#difine LINUX_IMP 

#define CHECK(call)                        \
{                                          \
  const cudaError_t error = call;          \
  if(error != cudaSuccess)                 \
  {\
  printf("Error: %s:%d,", __FILE__,__LINE__);\
  printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
  exit(1);\
	}\
}

//don't forget the time 
double cpuSecond() {

    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);

}

void checkResult(float *hostRef, float *gpuRef,const int N){
	double epsilon=1.0E-8;
	bool match = 1;
	for(int idx = 0; idx < N; idx++){
		if(abs(hostRef[idx]-gpuRef[idx]) > epsilon){
		match = 0;
		printf("Arrays do not match!\n");
		printf("host %5.6f gpu %5.6f at current %d\n",hostRef[idx],gpuRef[idx],idx);
		break;
		}
	}
	if(match)printf("Arrays match .\n\n");
}
//generate different seed for random number
void initiaData(float *ip,int size){
	time_t t;
	srand((unsigned)time(&t));
	try{
	    for(int i = 0; i < size; i++){
	       ip[i] = (float)(rand() & 0xFF)/10.0f;
	       //ip[i] = float(0.001*i);
	       //ip[i] = float(0.01);
		}
	}catch(...){
	printf("I don't know why!");
	}
}

void sumMatrixOnHost(float *A,float *B, float *C, const int nx, const int ny){
	for(int idy = 0; idy < ny; idy++)
        	for(int idx = 0; idx < nx; idx++){
        		C[idx+idy*nx] = A[idx+idy*nx] + B[idx+idy*nx];
    }

}


__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;
    if (ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}
/*****
//kernel function
__global__ void sumArrayOnGPU(float *A,float *B, float *C){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.x * blockDim.y;
	//printf("%d+%d*%d*%d = %d\n",idx,idy,blockDim.x,gridDim.x,idx+idy*blockDim.x*gridDim.x);
	C[idx + idy * blockDim.x * gridDim.x] = A[idx + idy * blockDim.x * gridDim.x] + B[idx + idy * blockDim.x * gridDim.x];
	//printf("%f\n",C[idx+idy*blockDim.x*gridDim.x]);
}
*****/
void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %ld.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %ld.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %ld.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0],    prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %ld.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %ld.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
    printf("\n\n\n");
}

int main(int argc, char **argv){
	printf("%s Starting...\n",argv[0]);
	double istart,iElaps;
	//get the cuda device count
	int count;
	cudaGetDeviceCount(&count);
	if(count == 0){
		fprintf(stderr, "There is no device.\n");
		exit(1);
	}
	//find the device >= 1.x
	int idxDevice = 0;
	for(int idxDevice = 0; idxDevice < count; ++idxDevice){
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop,idxDevice) == cudaSuccess)
			if(prop.major >= 1){
				printDeviceProp(prop);
				break;
			}
	}
	if(idxDevice == count){
		fprintf(stderr,"there is no device supporting CUDA 1.x. \n");
	}
	CHECK(cudaSetDevice(idxDevice))
	
	//set up data size of Matrix
	int nx = 1<< 12;
	int ny = 1<< 12;
	int nxy = nx*ny;
	size_t nBytes = nxy * sizeof(float);
	printf("Matrix size: %d\n", nxy);
	printf("Matrix volume: %zu\n",nBytes);
  
	//malloc host memory
	float *h_A,*h_B,*hostRef,*gpuRef;
	istart = cpuSecond();
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef  = (float *)malloc(nBytes);
	iElaps = cpuSecond() -istart;
	printf("malloc host memory:%lfs\n",iElaps);
  
	//initialize data at host side 
	istart = cpuSecond();
	initiaData(h_A, nxy);
	initiaData(h_B, nxy);	
    	iElaps = cpuSecond() - istart;
	memset(hostRef,0,nBytes);
	memset(gpuRef,0,nBytes);
	printf("initiaData spent time:%lfs\n",iElaps);
	
	//add Matrix at host side for result checks
	istart = cpuSecond();
	sumMatrixOnHost(h_A,h_B, hostRef, nx, ny);
	iElaps = cpuSecond() - istart;
 	printf("sumArrayOnHost spent time:%lfs\n",iElaps);
	
	//malloc device global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float **) &d_A,nBytes);
	cudaMalloc((float **) &d_B,nBytes);
	cudaMalloc((float **) &d_C,nBytes);
  
	//transfer data from host to device
	cudaMemcpy(d_A, h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B,nBytes,cudaMemcpyHostToDevice);
  
	//invoke kenel at host side
	int dimx = 32;
	int dimy = 1;
	dim3 block(dimx,dimy);
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
	//dim3 grid(512,512);

	istart = cpuSecond();
	sumMatrixOnGPU2D<<<grid, block>>>(d_A,d_B,d_C,nx,ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - istart;
	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %lf sec\n", grid.x,grid.y, block.x, block.y, iElaps);

	//copy kenel result back to host side
	cudaMemcpy(gpuRef, d_C,nBytes,cudaMemcpyDeviceToHost);
	//for(int i = ;i<nxy;i++){printf("%f",gpuRef[i])}

	//check device results
	checkResult(hostRef, gpuRef,nxy);
  
	//free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
  
	//free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
  
	//reset device
	cudaDeviceReset();
	return 0;
}

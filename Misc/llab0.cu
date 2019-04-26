#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>

#define CHECK(call)\
{\
  const cudaError_t error = call;\
  if(error != cudaSuccess)\
  {\
  printf("Error: %s:%d,", __FILE__,__LINE__)\
  printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
  exit(1);\
  }\
}

void checkResult(float *hostRef, float *gpuRef,const int N){
	double epsilon=1.0E-8;
	bool match = 1;
	for(int idx = 0; idx < N; idx++){
		if(abs(hostRef[idx]-gpuRef[idx]) > epsilon){
		match = 0;
		printf("Arrays do not match!\n");
		printf("host %5.2f gpu %5.2f at current %d\n",hostRef[idx],gpuRef[idx],idx);
		break;
		}
	}
	if(match)printf("Arrays match .\n\n");
}


//generate different seed for random number
void initiaData(float *ip,int size){
	time_t t;
	srand((unsigned)time(&t));
	for(int i = 0; i < size; i++){
		ip[i] = (float)(rand() & 0xFF)/10.0f;
	}
}

void sumArrayOnHost(float *A,float *B, float *C, const int N){
	for(int idx = 0; idx <N; idx++)
		C[idx] = A[idx] + B[idx];
}
__global__ void sumArrayOnGPU(float *A,float *B, float *C){
	int idx = threadIdx.x;
	C[idx] = A[idx]+B[idx];
}

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0],    prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
	printf("\n\n\n");
}

int main(int argc, char **argv){
	printf("%s Starting...\n",argv[0]);
	
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
	cudaSetDevice(idxDevice);
	
	//set up data size of vectors
	int nElm = 32;
	printf("Vector size %d\n", nElm);
	//malloc host memory
	size_t nBytes = nElm * sizeof(float);
	float *h_A,*h_B,*hostRef,*gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef  = (float *)malloc(nBytes);
	//initialize data at host side 
	initiaData(h_A, nElm);
	initiaData(h_B, nElm);	
	memset(hostRef,0,nBytes);
	memset(gpuRef,0,nBytes);
	//malloc device global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float **) &d_A,nBytes);
	cudaMalloc((float **) &d_B,nBytes);
	cudaMalloc((float **) &d_C,nBytes);	
	//transfer data from host to device
	cudaMemcpy(d_A, h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B,nBytes,cudaMemcpyHostToDevice);
	//invoke kenel at host side
	dim3 block(nElm);
	dim3 grid(nElm/block.x);
	sumArrayOnGPU<<< grid, block >>>(d_A,d_B,d_C);
	printf("Execution configuration <<<%d, %d>>>\n",grid.x,block.x);
	//copy kenel result back to host side
	cudaMemcpy(gpuRef, d_C,nBytes,cudaMemcpyDeviceToHost);
	//add vector at host side for result checks
	sumArrayOnHost(h_A,h_B, hostRef, nElm);
	//check device results
	checkResult(hostRef, gpuRef,nElm);
	//free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	//free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
		
	return 0;


}

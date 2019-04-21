#include <stdio.h>
#include <iostream>
#include "gputimer.h"
#include <device_launch_parameters.h>
#include <time.h>
#include <sys/time.h>
using namespace std;
const int BLOCK_SIZE = 16;
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
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct{
    int width;
    int height;
    float* elements;
} Matrix;

//Timer
double cpuSecond() {
    struct timeval tp;                        
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);

}
//generate different seed for random number
void initiaData(Matrix A){
	time_t t;
	srand((unsigned)time(&t));
	try{
	    for(int i = 0; i < A.height; i++){
		for(int j = 0; j<A.width; j++){
	       	    A.elements[i * A.width + j] = (float)(rand() & 0xFF)/10.0f;
		}
            }
	}catch(...){
	printf("I don't know why!");
	}
}

void cpu_MMKernel(const Matrix A, const Matrix B, Matrix C){
    for(int row = 0; row < A.height; row++){
        for(int col = 0; col < B.width; col++ ){
            for(int idx = 0; idx <C.width; idx++){
                C.elements[row * C.width + col] += A.elements[row * A.width + idx] * B.elements[idx * B.width + col];
            }
        }
    }
}

__global__
void global_MMKernel(const Matrix A, const Matrix B, Matrix C){
float Cvalue = 0;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
for(int e = 0; e < A.width; ++e){
    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    }
C.elements[row * C.width + col] = Cvalue;
}

__device__ float GetElement(const Matrix A, int row, int col){
    return A.elements[row * A.width + col]; 
}

__device__ void SetElement(Matrix A, int row, int col,float val){
    A.elements[row * A.width + col] = val;
}

__device__ Matrix GetSubMatrix(Matrix A,int row, int col){
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.elements = &A.elements[row * BLOCK_SIZE * A.width
			      +col * BLOCK_SIZE];
    return Asub;
}

__global__
void shmeme_MMKernel(const Matrix A, const Matrix B, Matrix C){
    //block row and col
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C,blockRow,blockCol);    
    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int m = 0; m < (A.width / BLOCK_SIZE); ++m){
        Matrix Asub = GetSubMatrix(A,blockRow,m);
        Matrix Bsub = GetSubMatrix(B,m,blockCol);
    
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
    
        __syncthreads();
	for(int e = 0; e < BLOCK_SIZE; e++){
            Cvalue += As[row][e] * Bs[e][col];
        }
	__syncthreads();
    }
    SetElement(Csub,row,col,Cvalue);
}

int main(int argc, char **argv){
    //init system
    printf("%s Starting...\n",argv[0]);
    double istart,iElaps;
    //get the cuda device count
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0){
    	fprintf(stderr, "There is no device.\n");
	exit(1);
    }
    //find the first device >= 1.x
    int idxDevice = 0;
    for(idxDevice = 0; idxDevice < count; ++idxDevice){
	cudaDeviceProp prop;
	if(cudaGetDeviceProperties(&prop,idxDevice) == cudaSuccess)
		if(prop.major >= 1){
			//printDeviceProp(prop);
			break;
		}
    }
    if(idxDevice == count){
        fprintf(stderr,"there is no device supporting CUDA 1.x. \n");
    }
    CHECK(cudaSetDevice(idxDevice))

    //set up data size
    Matrix h_A,h_B, h_C, hf_C;
    h_A.width = 1<<10;
    h_A.height = 1<<9;
    h_B.width = 1<<9;
    h_B.height = 1<<10;
    h_C.width = h_B.width;
    h_C.height = h_A.height;
    hf_C.width = h_B.width;
    hf_C.height = h_A.height;
    Matrix d_A,d_B,d_C;
    d_A.width = 1<<10;
    d_A.height = 1<<9;
    d_B.width = 1<<9;
    d_B.height = 1<<10;
    d_C.width = d_B.width;
    d_C.height = d_A.height;

    int Awh = h_A.width * h_A.height;
    int Bwh = h_B.width * h_B.height;
    int Cwh = h_C.width * h_C.height;
    size_t Abytes = Awh * sizeof(float);
    size_t Bbytes = Bwh * sizeof(float);
    size_t Cbytes = Cwh * sizeof(float);
    
    //malloc host memry
    h_A.elements  = (float *)malloc(Abytes);
    h_B.elements  = (float *)malloc(Bbytes);
    h_C.elements  = (float *)malloc(Cbytes);
    hf_C.elements = (float *)malloc(Cbytes);
 
    //init data
    initiaData(h_A);
    initiaData(h_B);
    
    //malloc device global memory
    cudaMalloc((float **) &d_A.elements, Abytes);
    cudaMalloc((float **) &d_B.elements, Bbytes);
    cudaMalloc((float **) &d_C.elements, Cbytes);
    
    //transfer data from host to device
    cudaMemcpy(d_A.elements, h_A.elements,Abytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, h_B.elements,Bbytes,cudaMemcpyHostToDevice);
    
    //invoke kernel at host side
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((h_C.width + block.x -1)/block.x ,(h_C.height + block.y - 1)/block.x);
 
    istart = cpuSecond();
    global_MMKernel<<<grid,block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("GPU_global_MMKernel <<<(%d,%d), (%d,%d)>>> elapsed %lf ms\n", grid.x,grid.y, block.x, block.y, 1000*iElaps);
   
    istart = cpuSecond();
    shmeme_MMKernel<<<grid,block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("GPU_shmem_MMKernel <<<(%d,%d), (%d,%d)>>> elapsed %lf ms\n", grid.x,grid.y, block.x, block.y, 1000*iElaps);

    //copy kenel result back to the host side 
    cudaMemcpy(h_C.elements, d_C.elements, Cbytes, cudaMemcpyDeviceToHost);
    
    
    istart = cpuSecond();
    cpu_MMKernel(h_A, h_B, hf_C);
    iElaps = cpuSecond() - istart;
    printf("CPU_MMKernel elapsed %lf ms\n",1000*iElaps);

    //free device global
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
    cudaFree(d_C.elements);

    //free host memory
    free(h_A.elements);
    free(h_B.elements);
    free(h_C.elements);
    free(hf_C.elements);

    //reset device
    cudaDeviceReset();
    return 0;
}

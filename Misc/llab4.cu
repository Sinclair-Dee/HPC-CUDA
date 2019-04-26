#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include<sys/time.h>

//don't forget the time 
double cpuSecond() {
//#ifdef  LINUX_IMP
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
//#endif
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

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;
    if (ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}


int main(int argc, char **argv){
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);
	//printf("%s using Device %d: %s\n", argv[0],dev, deviceProp.name);
	double iStart,iElaps;
	
	// set up data size;
	int nx = 1<<12;
	int ny = 1<<12;
	int nxy = nx*ny;
	size_t nBytes = nxy*sizeof(float);
	int dimx = 32, dimy = 32;
	if (argc > 2) {
    		dimx = atoi(argv[1]);
    		dimy = atoi(argv[2]);
	}
	
	//malloc host memory
	float *h_A,*h_B,*gpuRef;
	iStart = cpuSecond();
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	gpuRef  = (float *)malloc(nBytes);
	iElaps = cpuSecond() -iStart;
	//printf("malloc host memory:%lfs\n",iElaps);
  
	//initialize data at host side 
	iStart = cpuSecond();
	initiaData(h_A, nxy);
	initiaData(h_B, nxy);	
    	iElaps = cpuSecond() - iStart;
	//printf("initiaData spent time:%lfs\n",iElaps);
	
	//malloc device global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float **) &d_A,nBytes);
	cudaMalloc((float **) &d_B,nBytes);
	cudaMalloc((float **) &d_C,nBytes);
  
	//transfer data from host to device
	cudaMemcpy(d_A, h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B,nBytes,cudaMemcpyHostToDevice);
  
	// set up execution configuration
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	//printf("Execution Configure (block %d grid %d)\n",block.x, grid.x);

	//invoke kenel at host side
	iStart = cpuSecond();
	sumMatrixOnGPU2D<<<grid, block>>>(d_A,d_B,d_C,nx,ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %lf us\n", grid.x,grid.y, block.x, block.y, 1000000*iElaps);

	//copy kenel result back to host side
	cudaMemcpy(gpuRef, d_C,nBytes,cudaMemcpyDeviceToHost);
	//for(int i = ;i<nxy;i++){printf("%f",gpuRef[i])}

	//check device results
	//checkResult(hostRef, gpuRef,nxy);
  
	//free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
  
	//free host memory
	free(h_A);
	free(h_B);
	free(gpuRef);
  
	//reset device
	cudaDeviceReset();
	return 0;
	
}


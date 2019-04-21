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

__global__ void warmup(float *c){// Branch Efficiency = 100.00%
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
    	a = b = 0.0f;
    	c[tid] = a + b;
}

__global__ void mathKernel1(float *c){// Branch Efficiency = 83.3%
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c){// Branch Efficiency = 100.00%
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c) {// Branch Efficiency = 71.43%
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	bool ipred = (tid % 2 == 0); 
	if (ipred) {
		ia = 100.0f;
	}
	if (!ipred) {
		ib = 200.0f;
	}
	c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c) {// Branch Efficiency = 71.43%
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	bool ipred = (tid % 2 == 0);
	if (ipred) {
		ia = 100.0f;
	}
	if (!ipred) {
		ib = 200.0f;
	}
	c[tid] = ia + ib;
}



int main(int argc, char **argv){
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);
	printf("%s using Device %d: %s\n", argv[0],dev, deviceProp.name);
	double iStart,iElaps;
	
	// set up data size;
	int size = 64;
	int blocksize =64;
	if(argc > 1) blocksize = atoi(argv[1]);
	if(argc > 2) size = atoi(argv[2]);
	size_t nBytes = size*sizeof(float);
		
	// set up execution configuration
	dim3 block(blocksize,1);
	dim3 grid((size+block.x-1)/block.x,1);
	printf("Execution Configure (block %d grid %d)\n",block.x, grid.x);
	
	// allocate gpu memory
	float *d_C;
	cudaMalloc((float**)&d_C,nBytes);
	
	// run a warmup kernel to remove overhead;
	cudaDeviceSynchronize();
	iStart = cpuSecond();
	warmup<<<grid,block>>>(d_C);
	cudaDeviceSynchronize();
    	iElaps = cpuSecond() - iStart;
	printf("warmup <<< %4d %4d >>> elapsed %lf sec \n",grid.x,block.x, iElaps);
	
	// run kenel 1
	cudaDeviceSynchronize();
	iStart = cpuSecond();
	mathKernel1<<<grid, block>>>(d_C);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("mathKernel1 <<< %4d %4d >>> elapsed %lf sec \n",grid.x,block.x,iElaps );
	
	// run kernel 2
	iStart = cpuSecond();
	mathKernel2<<<grid, block>>>(d_C);
	cudaDeviceSynchronize();
	iElaps = cpuSecond () - iStart;
	printf("mathKernel2 <<< %4d %4d >>> elapsed %lf sec \n",grid.x,block.x,iElaps );
	
	// run kernel 3
	iStart = cpuSecond ();
	mathKernel3<<<grid, block>>>(d_C);
	cudaDeviceSynchronize();
	iElaps = cpuSecond () - iStart;
	printf("mathKernel3 <<< %4d %4d >>> elapsed %lf sec \n",grid.x,block.x,iElaps);
	
	// run kernel 4
	iStart = cpuSecond ();
	mathKernel4<<<grid, block>>>(d_C);
	cudaDeviceSynchronize();
	iElaps = cpuSecond () - iStart;
	printf("mathKernel4 <<< %4d %4d >>> elapsed %lf sec \n",grid.x,block.x,iElaps);
	
	// free gpu memory and reset divece
	cudaFree(d_C);
	cudaDeviceReset();
	return EXIT_SUCCESS;
	
}


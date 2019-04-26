#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, size_t size);

__global__ void addKernel(int *c, const int *a){

  int i = threadIdx.x;
  extern __shared__ int smem[];
  smem[i] = a[i];
  __syncthreads();
if(i==0)//0号线程做平方和
  {
	c[0] = 0;
	for(int d = 0; d<5;d++)
	{
		c[0]+=smem[d]*smem[d];
	}
  }

if(i == 1)//1号线程做累加
  { 
	c[1] = 0;
	for(int d = 0;d<5;d++)
	{
		c[1] += smem[d];
	}
  }
 if(i == 2)//2号线程做累乘
  {
	c[2] = 1;
	for(int d = 0;d<5;d++)
	{
		c[2] *= smem[d];
	}
  }
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    int c[arraySize] = { 0 };
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, arraySize);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	printf("\t1+2+3+4+5 = %d\n\t1^2+2^2+3^2+4^2+5^2 = %d\n\t1*2*3*4*5 = %d\n\n\n\n\n\n", c[1], c[0], c[2]);
    // cudaThreadExit must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    }
    return 0;
}
 
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a,  size_t size)
{
    int *dev_a = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
 
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
 
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
 
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, 5, size*sizeof(int), 0>>>(dev_c, dev_a);
    // cudaThreadSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
 
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
 
Error:
    cudaFree(dev_c);
    cudaFree(dev_a);    
    return cudaStatus;
}

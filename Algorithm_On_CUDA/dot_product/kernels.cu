#include "kernels.cuh"

__global__
void dot_product_kernel(float *x, float *y, float *dot, unsigned int n){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    __shared__ float cache[256];
    
    double temp = 0.0;
    while(index < n){
    temp += x[index] * y[index];
    index += stride; 
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    //reduction
    unsigned int i = blockDim.x/2;
    while(i !=0){
        if(threadIdx.x < i){
        cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    //atom operation
    if(threadIdx.x == 0){
    atomicAdd(dot,cache[0]);
    }
}


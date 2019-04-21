#include "kernels.cuh"

__global__ 
void find_maxnum_kernel(float* d_array, float* d_max, int* d_mutex, unsigned int N){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x*blockDim.x;
    unsigned int offset = 0;

    //share  memory
    __shared__ float cache[512];
    
    float temp = -1.0;
    while(index + offset < N){
        temp = fmaxf(temp,d_array[index + offset]); 
        offset += stride;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();
    
    //reduction
    unsigned int i = blockDim.x/2;
    while(i != 0){
	if(threadIdx.x < i){
            cache[threadIdx.x] = fmaxf(cache[threadIdx.x],cache[threadIdx.x + i]); 
        }
        __syncthreads();//在if外
        i /= 2;
    }
    if(threadIdx.x == 0){
        while(atomicCAS(d_mutex,0,1) != 0);//lock
        *d_max = fmaxf(*d_max,cache[0]);
        atomicExch(d_mutex,0);//unlock
    }
}


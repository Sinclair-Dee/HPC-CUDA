#include "kernels.cuh"
#include<stdio.h>

__global__ void init_primes_kernel(int *primes, unsigned int N){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;
    while(index + offset < N){
        primes[index + offset] = index +offset + 1;
        offset +=stride;
    }
}//end init primes_kenel

__global__ void sieve_of_eratosthenes_kernel(int *primes, unsigned int N, unsigned int sqrtRootN){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + 2;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;
    
//    __shared__ unsigned int cache[N];
    
    while(index + offset <= sqrtRootN){
        unsigned int temp = index + offset;
        for(unsigned int i = temp * temp; i < N; i += temp){
            primes[i-1] = 0;
        }
        offset += stride;
    }
}

//TO DO
__global__ void sieve_of_eratosthenes_kernel2(){

}

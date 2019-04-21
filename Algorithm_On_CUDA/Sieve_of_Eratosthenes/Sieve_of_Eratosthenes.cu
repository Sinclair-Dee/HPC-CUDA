#include <iostream>
#include<ctime>
#include "kernels.cuh"

int main(){
    unsigned int N = 1*1024*1024;
    unsigned int M = (unsigned int)sqrt(N);
    int *h_primes;
    int *d_primes;
    
    //allocate memory
    h_primes = (int*)malloc(N*sizeof(int));
    cudaMalloc((int **)&d_primes,N*sizeof(int));

    //timeing on GPU
    float gpu_elapsed_time;
    cudaEvent_t gpu_start,gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    
    //call kernel
    dim3 grid = 32;
    dim3 block = 32;
    //init primes form 1->N
    init_primes_kernel<<<grid,block>>>(d_primes,N);
    //Sieve of eratosthenes 
    sieve_of_eratosthenes_kernel<<<grid,block>>>(d_primes, N, M);

    //copy reslts back to host
    cudaMemcpy(h_primes, d_primes, N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaEventRecord(gpu_stop,0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    std::cout<<"GPU took: "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;

    //cpu version
    clock_t cpu_start = clock();
    for(unsigned int i = 0; i < N; i++){
        h_primes[i] = i + 1;
    }
    for(unsigned int i = 2; i <= M; i++){
        unsigned int start = i*i;
        for(unsigned int j = start; j <= N; j += i){
            h_primes[j-1] = 0; 
        }
    }
    clock_t cpu_stop = clock();
    clock_t cpu_elapsed_time = 1000 * (cpu_stop - cpu_start)/CLOCKS_PER_SEC;
    std::cout<<"The CPU took: "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;

    //free memory
    free(h_primes);
    cudaFree(d_primes); 
}



#include <iostream>
#include <time.h>
#include <random>
#include <curand.h>
#include <math.h>
#include "kernels.cuh"

int main(){
    unsigned int n = 256*256;
    unsigned int m = 1<<14;
    int *h_count;
    int *d_count;
    curandState *d_state;
    float pi;

    //allocate memory
    h_count = (int*)malloc(sizeof(int));
    cudaMalloc((int **)&d_count,sizeof(int));
    cudaMalloc((curandState **) &d_state, n*sizeof(curandState));
    cudaMemset(d_count,0,sizeof(int));
    //set up timing stuff
    float gpu_elapsed_time;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    //set kernel 
    dim3 grid = 256;
    dim3 block = 256;
    setup_kernel<<<grid,block>>>(d_state);
    //monte carlo kernel
    monte_carlo_pi_kernel<<<grid, block,256*sizeof(int)>>>(d_state, d_count, m);
    
    //copy results back to the host
    cudaMemcpy(h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost);
    
    cudaEventRecord(gpu_stop,0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    //display results and timings for GPU
    pi = (*h_count)*4.0/(n*m);
    std::cout<<"Approximate pi calculated on GPU is: "<<pi<<" and calculation took "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;    
    
    //serial version
    clock_t cpu_start = clock();
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0,1.0);
    unsigned int count = 0;
    for(unsigned int i = 0; i < n; i++){
        int temp = 0;
        while(temp < m){
            float x = distribution(generator);
            float y = distribution(generator);
            float rr = x*x +y*y;
            if(rr <= 1){
                count++;
            }
            temp++;
        }
    }
    clock_t cpu_stop = clock();
    pi = 4.0*count/(n*m);
    std::cout<<"Approximate pi calculated on CPU is: "<<pi<<" and calculation took "<<1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC<<" milli-seconds"<<std::endl;
    //free memory
}


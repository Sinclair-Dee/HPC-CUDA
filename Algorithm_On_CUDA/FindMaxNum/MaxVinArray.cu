#include<iostream>
#include<stdlib.h>
#include<cstdlib>
#include<ctime>
#include "kernels.cuh"

int main(){
    unsigned int N = 1024*1024*16;
    float *h_array;
    float *d_array;
    float *h_max;
    float *d_max;
    int *d_mutex;

    //allocate memory
    h_array = (float*)malloc(N*sizeof(float));
    h_max = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&d_array, N*sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));
    cudaMalloc((void**)&d_mutex, sizeof(int));
    cudaMemset(d_max, 0, sizeof(float));
    cudaMemset(d_mutex, 0, sizeof(float));
   
    //fill host arry with data
    for(int i = 0; i < N; i++){
        h_array[i] = N * float(rand())/RAND_MAX;
    }    

    // set up timing variables
    float gpu_elapsed_time;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);

    //copy from host to device
    cudaMemcpy(d_array, h_array, N*sizeof(float),cudaMemcpyHostToDevice);

    //call kenel
    dim3 block = 1<<9;
    dim3 grid = 1<<7;
    for(unsigned int j = 0; j < 1000; j++){
    find_maxnum_kernel<<<grid,block>>>(d_array, d_max, d_mutex, N);
    }

    //copy from device to host
    cudaMemcpy(h_max, d_max,sizeof(float),cudaMemcpyDeviceToHost);

    //timing report
    cudaEventRecord(gpu_stop,0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start,gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
   
    //report results
    std::cout<<"Maximum number found on gpu was: "<<*h_max<<std::endl;
    std::cout<<"The gpu took: "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;
 
    //run cpu version and report time
    clock_t cpu_start = clock();
    for(unsigned int j = 0; j < 1000; j++){
        *h_max = -1.0;
        for(unsigned int i = 0; i < N; i++){
            *h_max = h_array[i] > *h_max ? h_array[i] : *h_max;
        }
    }   
    clock_t cpu_stop = clock();
    clock_t cpu_elapsed_time = 1000 * (cpu_stop - cpu_start)/CLOCKS_PER_SEC;

    std::cout<<"Maximum number found on cpu was: "<<*h_max<<std::endl;
    std::cout<<"The cpu took: "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;
   
    //evolute result from CPU and GPU
    //TO DO
    //free memory
    free(h_array);
    free(h_max);
    cudaFree(d_array);
    cudaFree(d_max);
    cudaFree(d_mutex);
}


#include<iostream>
#include<stdlib.h>
#include<cstdlib>
#include<ctime>
#include<random>
#include "kernels.cuh"

int main(){
    unsigned int N = (1<<12)*256*256;
    float *h_prod;
    float *d_prod;
    float *h_x, *h_y;
    float *d_x, *d_y;

    //allocate memory
    h_prod = (float*)malloc(sizeof(float)); 
    h_x = (float*)malloc(N*sizeof(float));
    h_y = (float*)malloc(N*sizeof(float));
    cudaMalloc((void**)&d_prod, sizeof(float));
    cudaMalloc((void**)&d_x, N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(float));
    cudaMemset(d_prod, 0.0, sizeof(float));

    //fill host arry with data
    for(int i = 0; i < N; i++){
        h_x[i] = float(rand()%N)/N;
        h_y[i] = float(rand()%N)/N;
    }    
    // set up timing variables
    float gpu_elapsed_time = 0.0;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    //copy from host to device
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(gpu_start,0);
    
     //call kenel
    dim3 block = 1<<8;
    dim3 grid  = 1<<8;
    dot_product_kernel<<<grid,block>>>(d_x, d_y, d_prod, N);

    //time report 
    cudaEventRecord(gpu_stop,0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time,gpu_start,gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    
    //copy data back to host
    cudaMemcpy(h_prod, d_prod,sizeof(float),cudaMemcpyDeviceToHost);

    //report results
    std::cout<<"dot product computed on GPU is: "<<*h_prod<<" and took "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;
    
    //// run CPU based dot product to compare times to GPU code
    clock_t cpu_start = clock();
    double temp = 0.0;
    for(unsigned int j = 0; j < N; j++){
       temp += h_x[j] * h_y[j];   
    }
    *h_prod = temp;
    clock_t cpu_stop = clock();
    clock_t cpu_elapsed_time = 1000 * (cpu_stop - cpu_start)/CLOCKS_PER_SEC;
    std::cout<<"dot product computed on CPU is: "<<*h_prod<<" and took "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;
   
    //evolute result from CPU and GPU
    //TO DO

    //free memory
    free(h_prod);
    free(h_x);
    free(h_y);
    cudaFree(d_prod);
    cudaFree(d_x);
    cudaFree(d_y);
}


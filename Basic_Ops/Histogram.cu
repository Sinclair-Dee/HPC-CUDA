#include<stdio.h>
#include<iostream>
#include<ctime>
#include<cuda_runtime.h>

int log2(int i){
    int r = 0;
    while(i >>=1) r++;
    return r;
}

int bit_reverse(int w, int bits){
    int r = 0;
    for(int i = 0; i < bits;i++){
        int bit = (w & (1 << i))>>i;//从低位起的第i位
        r |= bit << (bits - i - 1); //低位转为高位
    }
    return r;
}


void cpu_histo(int *h_bins, int *h_in, const int ARRAY_SIZE, const int BIN_COUNT){
    for(int i = 0; i < ARRAY_SIZE; i++){
        h_bins[h_in[i]%BIN_COUNT] += 1;
    }
}


__global__ void simple_histo(int *d_bins,const int *d_in,const int BIN_COUNT){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[index];
    int myBin = myItem % BIN_COUNT;
    atomicAdd(&d_bins[myBin],1);
}

__global__ void local_histo(int *d_bins, const int *d_in, const int N, const int BIN_COUNT){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;//gridDim.x = 1
    int offset = 0;
    __shared__  int cache[512][16];
    for(int i = 0; i < BIN_COUNT; i++){
        cache[index][i] = 0;
    }
    __syncthreads();
    while(index + offset < N){
        int myItem = d_in[index + offset];
        int myBin = myItem%BIN_COUNT;
        cache[index][myBin] += 1;
        offset += stride;
    }
    __syncthreads();

    //reduction
    int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){
            for(int j = 0; j < BIN_COUNT; j++)
                cache[threadIdx.x][j] += cache[threadIdx.x + i][j]; 
        }
        __syncthreads();
        i /= 2;
    }
    if(threadIdx.x == 0){
        for(int i = 0; i < BIN_COUNT; i++)
            d_bins[i] = cache[0][i];
    }
}

__global__ void local_histo_op(int *d_bins, const int *d_in, const int N, const int BIN_COUNT){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    __shared__  int cache[16];
    for(int i = 0; i < BIN_COUNT; i++){
        cache[i] = 0;
    }
    __syncthreads();

    while(index + offset < N){
        int myItem = d_in[index + offset];
        int myBin = myItem%BIN_COUNT;
        atomicAdd(&cache[myBin],1);
        offset += stride;
    }
    __syncthreads();//此处的同步比较重要，确保每个block的cache完全获取后再写回 global memory.
    if(threadIdx.x < BIN_COUNT){
        atomicAdd(&d_bins[threadIdx.x],cache[threadIdx.x]);
    }
}


int main(int argc, char **argv){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        fprintf(stderr,"error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }    
    int dev = 0;
    cudaSetDevice(dev);
    
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }
    const int ARRAY_SIZE = 1<<20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    const int BIN_COUNT = 16;
    const int BIN_BYTES = BIN_COUNT * sizeof(int);

    //generate the input array on the host
    int h_in[ARRAY_SIZE];
    for(int i =0; i < ARRAY_SIZE; i++){
        h_in[i] = bit_reverse(i,log2(ARRAY_SIZE));
    }
    int h_bins[BIN_COUNT];
    for(int i = 0; i < BIN_COUNT; i++ ){
        h_bins[i] = 0;
    }
   
    //declare GPU memory pointers
    int *d_in;
    int *d_bins;

    //allocate GPU memory
    cudaMalloc((int **)&d_in,ARRAY_BYTES);
    cudaMalloc((int **)&d_bins,BIN_BYTES);

    // transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice); 

    //timing on GPU
    float gpu_elapsed_time;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);

    int whichKernel = 0;
    if(argc == 2){
        whichKernel = atoi(argv[1]);
    }

    // launch the kernel
    switch(whichKernel) {
    case 0:
        printf("Running simple histo\n");
        simple_histo<<<ARRAY_SIZE / 64, 64>>>(d_bins, d_in, BIN_COUNT);
        break;
    case 1:
        printf("Running local histo\n");
        local_histo<<<1,512>>>(d_bins, d_in,ARRAY_SIZE,BIN_COUNT);
        break;
    case 2:
        printf("Running local opti histo");
        local_histo_op<<<ARRAY_SIZE / 256, 256>>>(d_bins,d_in,ARRAY_SIZE,BIN_COUNT);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }
    // timing report
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    std::cout<<"GPU took: "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;

    // copy back the sum from GPU
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < BIN_COUNT; i++) {
        printf("GPU bin %d: count %d\n", i, h_bins[i]);
    }
    memset(h_bins, 0, BIN_COUNT * sizeof(int));
    
    clock_t cpu_start = clock();
    for(int i = 0; i < ARRAY_SIZE; i++){
        h_bins[h_in[i]%BIN_COUNT] += 1;
    }
    clock_t cpu_stop = clock();
    clock_t cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
    std::cout<<"The cpu took: "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;

    for(int i = 0; i < BIN_COUNT; i++) {
        printf("CPU bin %d: count %d\n", i, h_bins[i]);
    }

    //free GPU memory allocate
    cudaFree(d_in);
    cudaFree(d_bins);

    return 0;
}

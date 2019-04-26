#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
__global__ 
void global_scan(float* d_out,float* d_in){
    int idx = threadIdx.x;
    float out = 0.00f;
    d_out[idx] = d_in[idx];
    __syncthreads();
    for(int interpre = 1;interpre<sizeof(d_in);interpre *=2){
        if(idx-interpre >= 0){//从interpre开始加
	    //要先缓存到out中，否则本层的d_out可能会干扰。都完成了在传到下一层
	    out = d_out[idx] + d_out[idx-interpre];
	}
	__syncthreads();
	if(idx-interpre >= 0){
	d_out[idx] = out;
	out = 0.00f;	
	}
	__syncthreads();
    }
}

extern __shared__ float shmem[];//share memory 要动态分配。所以前边加extern.
__global__ 
void shmem_scan(float* d_out,float* d_in){
   // extern __shared__ float shmem[];
    float out = 0.00f;
    int idx = threadIdx.x;
    shmem[idx] = d_in[idx];
    __syncthreads();

    for(int interpre = 1; interpre < sizeof(d_in); interpre *= 2){
        if(idx-interpre >= 0){
        out = shmem[idx] + shmem[idx-interpre];
        }
        __syncthreads();
        if(idx-interpre >= 0){
        shmem[idx] = out;
        out = 0.00f; 
        }
	__syncthreads();//经实测，这里的同步必须要加上
    }
    //__syncthreads();//可加可不加
    d_out[idx] = shmem[idx];
}


int main(int argc,char** argv){
  const int ARRAY_SIZE = 8;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // generate the input array on the host
  float h_in[ARRAY_SIZE];
  for(int i=0;i<ARRAY_SIZE;i++){
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];

  // declare GPU memory pointers
  float* d_in;
  float* d_out;

  // allocate GPU memory
  cudaMalloc((void**) &d_in,ARRAY_BYTES);
  cudaMalloc((void**) &d_out,ARRAY_BYTES);

  // transfer the array to GPU
  cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);

  // launch the kernel
  shmem_scan<<<1,ARRAY_SIZE,ARRAY_SIZE*sizeof(float)>>>(d_out,d_in);

  // copy back the result array to the GPU
  cudaMemcpy(h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost);

  // print out the resulting array
  for(int i=0;i<ARRAY_SIZE;i++){
    printf("%f",h_out[i]);
    printf(((i%4) != 3) ? "\t" : "\n");
  }

  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;


}


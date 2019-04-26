#include <stdio.h>
#include "gputimer.h"

const int K32 = 32;
const int K16 = 16;
// Utility functions: compare, print, and fill matrices
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at: %s : %d\n", file,line);
    fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
    exit(1);
  }
}

bool compare_matrices(float *gpu, float *ref, const int N){
  for(int j = 0; j < N; j++){
    for(int i = 0; i < N; i++){
        if(abs(ref[i + j*N] - gpu[i + j*N]) > 0.0001){
            return true;
        }
    }
  }
  return false;
}

void print_matrix(float *mat, const int N) {	
  for(int j=0; j < N; j++) {
    for(int i=0; i < N; i++) { 
      printf("%4.4g ", mat[i + j*N]); 
      }
    printf("\n");
  }	
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat, const int N){
  for(int j = 0; j < N * N ; j++)
    mat[j] = static_cast<float>(j);
}

void transpose_CPU(float *in, float *out, const int N){
  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
      out[j + i * N] = in[i + j * N];//out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ 
void transpose_serial(float *in, float *out, const int N){
  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
      out[j + i * N] = in[i + j * N];//out(j,i) = in(i,j)
}

// to be launched with one thread per row of output matrix
__global__ 
void transpose_parallel_per_row(float *in, float *out, const int N){
  int i = threadIdx.x;
  
  for(int j=0; j < N; j++)
	out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per element, in KxK threadblocks
// thread (x,y) in grid writes element (i,j) of output matrix 
__global__ 
void transpose_parallel_per_element(float *in, float *out, const int N, const int K){
  int i = blockIdx.x * K + threadIdx.x;//每个block分配K*K个线程
  int j = blockIdx.y * K + threadIdx.y;//每个grid分配N/K * N/K个block.
  
  out[j + i * N] = in[i + j * N];
}

// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ 
void transpose_parallel_per_element_32_32_tiled(float *in, float *out, const int N, const int K){
  // (i,j) locations of the tile corners for input & output matrices:
  int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;
  int x = threadIdx.x, y = threadIdx.y;
  
  __shared__ float tile[K32][K32];
  tile[y][x] = in[in_corner_i + x + (in_corner_j+y) * N];
  __syncthreads();
  out[out_corner_i + y + (out_corner_j + x) * N] = tile[y][x];
}


__global__
void transpose_parallel_per_element_16_16_tiled(float *in, float *out, const int N, const int K){
  // (i,j) locations of the tile corners for input & output matrices:
  int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;
  int x = threadIdx.x, y = threadIdx.y;

  __shared__ float tile[K16][K16];
  tile[y][x] = in[in_corner_i + x + (in_corner_j+y) * N];
  __syncthreads();
  out[out_corner_i + y + (out_corner_j + x) * N] = tile[y][x];
}

// to be launched with one thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array padded to avoid bank conflicts
__global__ 
void transpose_parallel_per_element_32_32_tiled_padded1(float *in, float *out, const int N, const int K){
  // (i,j) locations of the tile corners for input & output matrices:
  int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

  int x = threadIdx.x, y = threadIdx.y;

  __shared__ float tile[K32][K32+1];
  // coalesced read from global mem, TRANSPOSED write into shared mem:
  tile[y][x] = in[in_corner_i + x + (in_corner_j + y) * N];
  __syncthreads();
  // read from shared mem, coalesced write to global mem:
  out[out_corner_i + y + (out_corner_j + x)*N] = tile[y][x];
}

__global__
void transpose_parallel_per_element_16_16_tiled_padded1(float *in, float *out, const int N, const int K){
  // (i,j) locations of the tile corners for input & output matrices:
  int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

  int x = threadIdx.x, y = threadIdx.y;

  __shared__ float tile[K16][K16+1];
  // coalesced read from global mem, TRANSPOSED write into shared mem:
  tile[y][x] = in[in_corner_i + x + (in_corner_j + y) * N];
  __syncthreads();
  // read from shared mem, coalesced write to global mem:
  out[out_corner_i + y + (out_corner_j + x)*N] = tile[y][x];
}

int main(int argc, char **argv){  

  //set up date
  const int N = 1024;
  int K = 32;
  const size_t numbytes = N * N * sizeof(float);

  //MALLOC host memory
  float *in =   (float *) malloc(numbytes);
  float *out =  (float *) malloc(numbytes);
  float *gold = (float *) malloc(numbytes);

  //init data and get the gold
  fill_matrix(in, N);
  transpose_CPU(in, gold, N);
  
  //MALLOC device memory  
  float *d_in, *d_out;
  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);
  cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

  GpuTimer timer;  
  // Now time each kernel and verify that it produces the correct result.
  timer.Start();
  transpose_serial<<<1,1>>>(d_in, d_out, N);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");
  
  timer.Start();
  transpose_parallel_per_row<<<1,N>>>(d_in, d_out, N);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_parallel_per_row: %g ms.\nVerifying transpose...%s\n", timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");
  
  dim3 block(K,K);
  dim3 grid(N/K, N/K);

  timer.Start();
  transpose_parallel_per_element<<<grid, block>>>(d_in, d_out, N, K);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_parallel_per_element: %g ms.\nVerifying transpose...%s\n", timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");
  
  timer.Start();
  transpose_parallel_per_element_32_32_tiled<<<grid, block, K*K*sizeof(float)>>>(d_in, d_out,N, K);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

  timer.Start();
  transpose_parallel_per_element_32_32_tiled_padded1<<<grid,block, K*K*sizeof(float)>>>(d_in, d_out, N, K);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_parallel_per_element_tiled_padded %dx%d: %g ms.\nVerifying...%s\n", K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success"); 
 
  K = 16;
  dim3 block16(K, K);
  dim3 grid16(N/K, N/K);

  timer.Start();
  transpose_parallel_per_element_16_16_tiled<<<grid16, block16,K*K*sizeof(float)>>>(d_in, d_out, N, K); 
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");
	
  timer.Start();
  transpose_parallel_per_element_16_16_tiled_padded1<<<grid16, block16, K*K*sizeof(float)>>>(d_in, d_out, N, K);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("transpose_parallel_per_element_tiled_padded %dx%d: %g ms.\nVerifying...%s\n", K, K,  timer.Elapsed(), compare_matrices(out, gold,N) ? "Failed" : "Success");

  //free data
  free(in);
  free(out);
  free(gold);
  cudaFree(d_in);
  cudaFree(d_out);
}

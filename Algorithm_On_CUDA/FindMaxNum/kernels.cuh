#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include<cuda_runtime.h>

__global__ void find_maxnum_kernel(float* d_array, float* d_max, int* d_mutex, unsigned int n);
#endif

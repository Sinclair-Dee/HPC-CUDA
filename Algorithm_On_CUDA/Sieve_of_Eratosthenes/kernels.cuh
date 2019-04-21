#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__


__global__ void init_primes_kernel(int *prime, unsigned int n);
__global__ void sieve_of_eratosthenes_kernel(int *prime, unsigned int n, unsigned int sqrRootN);
__global__ void sieve_of_eratosthenes_kernel2(int *prime,unsigned int n);

#endif



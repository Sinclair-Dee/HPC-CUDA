#include "./common/book.h"
#include "./common/image.h"

#define DIM 1000

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex( float a, float b):r(a),i(b) {}
    __device__ float magnitude2(void){ return r*r + i*i; }
 
    __device__ cuComplex operator*(const cuComplex &a){
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
     }
    __device__ cuComplex operator+(const cuComplex &a){
        return cuComplex(r+a.r, i+a.i);
    }
};




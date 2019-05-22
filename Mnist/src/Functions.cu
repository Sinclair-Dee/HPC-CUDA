#include "Net.h"

void clear(){
  usleep(100000);
  for(int i = 0; i < 100; i++) printf("\n")
}

int reverse_int32(int i){
  unsigned char byte1,byte2,byte3,byte4;
  byte1 = i&MAXBYTE;
  byte2 = (i>>8) &MAXBYTE;
  byte3 = (i>>16)&MAXBYTE;
  byte4 = (i>>24)&MAXBYTE;
  return ((int)byte1<<24) + ((int)byte2<<16) + ((int) byte3<<8) + (int)byte4;
}
namespace GPU_Scope{
//read [number of image]×28×28 MNIST data from {datapath}
//store data into the given float array
void read_data(const char* datapath, host_vector< host_vector<float> >& data){
  ifstream infile(datapath, ios::binary);
  if(!infile.isopen()){
    printf("FAILED TO OPEN FILE: %sn",datapath);
    return;
  }
  cout<<"==Input test image file: "<<datapath<<endl;
  //read the header information
  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;

  infile.read((char*)&magic_number,sizeof(magic_number));
  magic = reverse_int32(magic_number);
  cout<<"magic number: "<<magic_number << endl;

  infile.read((char*)&number_of_images,sizeof(number_of_images));
  number_of_images = reverse_int32(number_of_images);
  cout<<"number of images: "<<number_of_images<<endl;
}


__global__
void GEMM(float *W, float* Unroll_FM_in, float* FM_out,
          int M_height_in, int M_width_N_height_in, int N_width_in,
          int height_out, int width_out){
  __shared__ float shmemM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shmemN[TILE_WIDTH][TILE_WIDTH];
  int threadidx = threadIdx.x;
  int threadidy = threadIdx.y;
  int blockidx  = blockIdx.x;
  int blockidy  = blockIdx.y;
  int row = blockidy * TILE_WIDTH + threadidy;
  int col = blockidx * TILE_WIDTH + threadIdx;

  float Pvalue = 0;
  for(int m = 0; m < M_width_N_height_in; m++){
    if(row < M_height_in && (m * TILE_WIDTH + threadidx) < M_width_N_height_in)
      shmemM[threadidy][threadidx] = Unroll_FM_in[row * M_width_N_height_in + (m*TILE_WIDTH) + threadIdx];
    else
      shmemM[threadidy][threadidx] = 0;
    if((m*TILE_WIDTH + threadidy) < M_width_N_height_in && col < N_width_in)
      shmemN[threadidy][threadidx] = W[(m*TILE_WIDTH + threadidy) * M_width_N_height_in +col];
    else
      shmemN[threadidy][threadidx] = 0;
    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; k++){
      Pvalue + shmemM[threadidy][k] * shmemN[k][threadidx];
    }
    __syncthreads();
  }

  if(row < height_out && col < width_out){
      FM_out[row * width_out + col] = Pvalue;
  }
}

}

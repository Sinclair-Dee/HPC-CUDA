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

//CH_in:number of input feature maps
//CH_out: number of output feature maps
//H_in：height of each input image
//W_in：width of each input map image
//W_h_w：height (and width) of each filter bank
//FM_in：input feature maps
//W： convolution filters
//FM_out：output feature maps
//void Convolution::ConvLayerForward(int BATCH, host_vector<float>& FM_in,  int CH_in, int H_in, int W_in,
//                                   host_vector<float>& W, int W_h_w,  host_vector<float>& FM_out, int CH_out)
void ConvLayerForward(int batch, float *FM_in, int CH_in, int H_in, int W_in
                        float *W, int W_h_w, float *FM_out, int CH_out){
  int H_out = H_in - W_h_w - 1;
  int W_out = H_in - W_h_w - 1;
  for(int batch = 0; batch < BATCH ; batch++){//for each sample in the mini-batch
    for(int ch_out = 0; ch_out < CH_out; ch_out++){// for each output feature map
      for(int h = 0; h < H_out; h++){   // for each output element
        for(int w = 0; w < W_out; w++){ //of H_out * W_out size outputimage
          //h and w is not center point, it's upper left corner point of Input image
          FM_out[batch*(CH_out * H_out * W_out)+ch_out * (H_out * W_out) + h * W_out + w] = 0;
          for(int ch_in = 0; ch_in < CH_in; ch_in){
            for(int i = 0; i < W_h_w; i++){//h of filter
              for(int j = 0; j < W_h_w; j++){//w of filter
                FM_out[batch*(CH_out * H_out * W_out)+ch_out * (H_out * W_out) + h * W_out + w] +=
                         FM_in[batch*(CH_in*H_in*W_in) + ch_in*(H_in*W_in)+(h+i)*W_h_w + (w+j)] *
                         W[ch_out*(CH_in*W_h_w*W_h_w) + ch_in*(W_h_w*W_h_w)+i*W_h_w + j];
              }
            }
          }
        }
      }
    }
  }
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
__global__
void ConvLayerForwardGPUnaive(float *FM_in, float  *W, float *FM_out,
               int CH_in, int H_in, int W_in, int W_out, int W_h_w, int CH_out){
  int H_out = H_in - W_h_w +1;
  int NumTile = ceilf((float)W_out/TILE_WIDTH);
  NumTile = NumTile == 0 ? 1 : NumTile;
  int batch, ch_out, h, w, ch_in,i,j;
  batch = blockIdx.x;
  ch_out = blockIdx.y;
  h = (blockIdx.z / NumTile) * TILE_WIDTH + threadIdx.y; //h of H_out
  w = (blockIdx.z % NumTile) * TILE_WIDTH + threadIdx.x; //w of W_out

  float ResAcc = 0;
  //allocate each outpt point to one thread
  for(ch_in = 0; ch_in < CH_in; ch_in++){
    for(i = 0; i < W_h_w; i++){
      for(j = 0; j < W_h_w; j++){
        if(h < H_out && w < W_out){
          ResAcc += FM_in[batch*(CH_in*H_in*W_in) + ch_in*(H_in*W_in)+(h+i)*W_h_w + (w+j)] *
                    W[ch_out*(CH_in*W_h_w*W_h_w) + ch_in*(W_h_w*W_h_w)+i*W_h_w + j];

        }
      }
    }
  }
  if(h < H_out && w < H_out){
    FM_out[batch*(CH_out * H_out * W_out) + ch_out * (H_out * W_out) + h * W_out * w] = ResAcc;
  }
  #ifdef DEBUG_CONV_
    if(batch == 7 && ch_out == 5 && w == 27) printf("h = %d, w = %d , ResAcc = %.3lf\n",h,w,ResAcc)
  #endif
}
__global__
void ConvLayerForwardGPUtiled(float *FM_in, float  *W, float *FM_out,
               int CH_in, int H_in, int W_in, int W_out, int W_h_w, int CH_out){

  int batch, ch_out, h, w,h_base,w_base,h0,w0;
  int FM_tile_width = TILE_WIDTH + W_h_w - 1;
  int H_out = H_in - W_h_w + 1;
  int NumTile = ceilf((float)W_out/TILE_WIDTH);
  NumTile = NumTile == 0 ? 1 : NumTile;

  __shared__ float shmem[(TILE_WIDTH + CONV_KERNEL_SIZE -1)*
                         (TILE_WIDTH + CONV_KERNEL_SIZE -1)+
                         (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)]; //kernel size equal 5
  float *FM_shared =(float*)&shmem[0]; //
  float *W_shared = (float*)&shmem[FM_tile_width * FM_tile_with];

  batch = blockIdx.x;
  ch_out = blockIdx.y

  h_base = (blockIdx.z / NumTile)
  w_base = (blockIdx.z % NumTile)

  h0 = threadIdx.y;
  w0 = threadIdx.x;

  h = h_base * TILE_WIDTH + h0;
  w = w_base * TILE_WIDTH + w0;

  int ch_in, i, j;

  float ResAcc = 0;

  for(ch_in = 0; ch_in < CH_in; ch_in++){
    //LOAD weights for W[batch,ch_out..]
    if((h0 < W_h_w) && (w0 < W_h_w)){
      W_shared[h0 * (CONV_KERNEL) + w0] = W[ch_out*(CH_in*W_h_w*W_h_w) + ch_in*(W_h_w*W_h_w) + h0 * W_h_w + w0];
    }
    __syncthreads();

    for(int m = h; m < h_base * TILE_WIDTH + FM_tile_width; m += TILE_WIDTH){
      for(int n = w; n < w_based * TILE_WIDTH + FM_tile_width; n += TILE_WIDTH){
        if(m - h_base < FM_tile_width && (n-w_base) < FM_tile_width){
          FM_shared[(m - h_base)*FM_tile_width +(n-w_base)] =
                             FM_in[batch * (CH_in * H_in * W_in) + ch_in * (H_in*W_in)+m*W_in + n];
        }
      }
    }
    __syncthreads();

    for(i = 0; i < W_h_w; i++){
      for(j = 0; j < W_h_w; j++){
        if(h < H_out && w < W_out){
          ResAcc += FM_shared[(h0 + i)*X_tile_width + (w0 + j)] * W_shared[i * W_h_w + j];
        }
      }
    }
    __syncthreads();
    if(h < H_out && w < W_out){
      FM_out[batch *(CH_out*H_out*W_out)+ch_out*(H_out*W_out) + h * W_out + w] = ResAcc;
    }
  }
}
__global__
void unroll_Kernel(int CH_in, int H_in, int W_in, int W_h_w, float *FM_in, float *Unroll_FM_in ){
 // this->device_FM_in.resize(Mini_Batch * Inputimage_channel * Inputimage_height * Inputimage_width, 0);
 // this->device_Unroll_FM.resize((Inputimage_channel * W_width_height * W_width_height) * (Outputimage_width *Outputimage_height),0);

  int unroll_ch_in, unroll_fm, h_out, w_out, unroll_ch_base, i, j;
  int ThreadIdx = blockIdx.x * 1024 + threadIdx.x
  int H_out = h_in - W_h_w + 1;
  int W_out = w_in - W_h_w + 1;
  int Unroll_H_W = H_out * W_out;

  if(ThreadIdx < CH_in * Unroll_H_W){
    unroll_ch_in = ThreadIdx / Unroll_H_W;
    unroll_fm = ThreadIdx % Unroll_H_W
    h_out = unroll_fm / W_out;
    w_out = unroll_fm % W_out;
    unroll_ch_base = unroll_ch_in * W_h_w * W_h_w;

    for(i = 0; i< W_h_w; i++){
      for(j = 0; j < W_h_w; j++){
        h_unroll = unroll_base + i * W_h_w + j;
        if(unroll_ch_in < CH_in && (h_out + i < H_in) && (w_out + j) < W_out){
          Unroll_FM_in[h_unroll * Unroll_H_W + unroll_fm ] = FM_in[ch_in*(W_in * H_in) + (h_out + i) * W_in + (w_out + j)];
        }
      }
    }
  }
}

}

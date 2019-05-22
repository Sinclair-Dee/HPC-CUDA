#include "Net.h"

namespace GPU_Scope{

void Convolution::init(int minib, int Inputimage_h, int Inputimage_w, int Inputimage_ch, int W_w_h, int W_ch){
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,0.1);

  this->MiniBatch = minib;

  this->W_weight_height = W_w_h;
  this->W_channel = W_ch;//一维表达

  this->Inputimage_width = Inputimage_w;
  this->Inputimage_height = Inputimage_h;
  this->Inputimage_channel = Inputimage_ch;
  this->FM_in__width = Inputimage_ch * Inputimage_w * Inputimage_h;
  this->FM_in__height = minib;

  this->Outputimage_width=(Inputimage_width-W_width_height+1);
  this->Outputimage_height=(Inputimage_height-W_width_height+1);
  this->Outputimage_channel=W_channel/Inputimage_channel;
  this->FM_out_weight = Outputimage_channel * Outputimage_height * Outputimage_width;
  this->FM_out_height = minib;

  this->Unroll_FM_width  = Outputimage_width * Outputimage_height;
  this->Unroll_FM_height = Inputimage_channel * W_width_height * W_width_height;

  //allocate memory
  this->device_FM_in.resize(Mini_Batch * Inputimage_channel * Inputimage_height * Inputimage_width, 0);
  this->host_FM_in.resize(Mini_Batch * Inputimage_channel * Inputimage_height * Inputimage_width, 0);
  this->device_Unroll_FM.resize((Inputimage_channel * W_width_height * W_width_height) * (Outputimage_width *Outputimage_height),0);
  this->device_Unroll_FMT.resize((Outputimage_width*Outputimage_height) * (Inputimage_channel*W_width_height*W_width_height),0);
  this->host_Unroll_FM.resize((Inputimage_channel*W_width_height*W_width_height) * (Outputimage_width*Outputimage_height),0);
  this->host_W.resize(W_channel * W_width_height * W_width_height, 0.5);
  this->device_W.resize(W_channel * W_width_height * W_width_height, 0.5);
  this->device_WT.resize(W_channel * W_width_height * Outputimage_channel, 0.5);
  for(int i = 0; i < W_channel*W_width_height*W_width_height; i++){this->host_W[i] =  distribution(generator)};
  for(int i = 0; i < W_channel*W_width_height*W_width_height; i++){this->device_W[i] = distribution(generator)};
  this->host_FM_out.resize(MiniBatch * Outputimage_channel * Outputimage_width * Outputimage_height, 0);
  this->device_FM_out.resize(MiniBatch * Outputimage_channel * Outputimage_width * Outputimage_height, 0);
  this->host_Wgrad.resize(Outputimage_channel * Inputimage_channel * W_width_height * W_width_height, 0);
  this->device_Wgrad.resize(Outputimage_channel * Inputimage_channel * W_width_height * W_width_height, 0);
  this->device_Wgrad_Temp.resize(Outputimage_channel * Inputimage_channel * W_width_height * W_width_height, 0)
}

void Convlution::forward_CPU(){
  float *input_pointer = thrust::raw_pointer_cast(host_FM_in,data());
  float *W_pointer = thrust::raw_pointer_cast(host_W.data());
  float *Output_pointer = thrust::raw_pointer_cast(host_FM_out()_)
  ConvLayerForward(MiniBatch,input_pointer,Inputimage_channel, Inputimage_height,Inputimage_width, W_pointer, W_width_height, Output_pointer, Outputimage_channel);
}

void Convlution::forward_CPU_naive(){
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  int bz = ceil((float)Outputimage_width/TILE_WIDTH) * ceil((float)Outputimage_height/TILE_WIDTH);
  bz = bz == 0 ? 1 : bz
  dim3 nunBlocks(MiniBatch, Outputimage_channel , bz);

  float *input_pointer = thrust::raw_pointer_cast(device_FM_in.data());
  float *W_pointer = thrust::raw_pointer_cast(device_W.data());
  float *Output_pointer = thrust::raw_pointer_cast(device_FM_out.data());

  ConvLayerForwardGPUnaive<<<numBlocks, threadsPerBlock>>>(input_pointer, W_pointer, Output_pointer,
                                  Inputimage_channel, Inputimage_height, Inputimage_width , Outputimage_width, W_width_height, Outputimage_channel);


}

void Convlution::forward_GPU_tiled(){
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  int bz = ceil((float)Outputimage_width / TILE_WIDTH) * ceil((float)Outputimage_height / TILE_height);
  bz = bz == 0 ? 1 :bz;
  dim3 numBlocks(MiniBatch, Outputimage_channel, bz);

  float *input_pointer = thrust::raw_pointer_cast(device_FM_in.data());
  float *W_pointer = thrust::raw_pointer_cast(device_W.data());
  float *Output_point = thrust::raw_pointer_cast(device_FM_out.data());

  ConvLayerForwardGPUtiled<<<mumBlocks, threadsPerBlock>>>(input_pointer, W_pointer, Output_pointer,
			Inputimage_channel, Inputimage_height, Inputimage_width , Outputimage_width, W_width_height, Outputimage_channel);

}

void Convlution::forward_GPU_gemm(){
  float* Output_pointer = thrust::raw_pointer_cast( device_FM_out.data() );
  float* FM_in__pointer = thrust::raw_pointer_cast( device_FM_in.data() );
  float* Unroll_FM_in_pointer = thrust::raw_pointer_cast( device_Unroll_FM.data() );

  for(int i = 0; i< MiniBatch; i++){
    int H_out = Inputimage_height - W_h_w + 1;
    int W_out = Inputimage_width - W_h_w + 1;

    float *batchFM_in_pointer = FM_in_pointer + i * (Inputimage_channel * Inputimage_height * Inputimage_width);
    float *

    int num_Thread = Inputimage_channel * Outputimage_height*Outputimage_width;
    int num_Blocks = ceil((float)num_Thread/1024);

    unroll_Kernel<<<num_blocks, 1024>>>(Inputimage_channel, Inputimage_height,Inputimage_width, W_width_height,
                                        FM_in_pointer, Unroll_FM_in_pointer);

    float *W_pointer = thust::raw_pointer_cast(device_W.data());




  }
}

void Convlution::backward_GPU_gemm(){

}

void Convolution::convLayer_forward(){

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
void ConvLayerForward(int BATCH, float *FM_in, int CH_in, int H_in, int W_in
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


//

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
void unroll_Kernel(int CH_in, int H_in, int W_in, int W_h_w, float *FM_in, float *Unroll_FM_in );
 // this->device_FM_in.resize(Mini_Batch * Inputimage_channel * Inputimage_height * Inputimage_width, 0);
 // this->device_Unroll_FM.resize((Inputimage_channel * W_width_height * W_width_height) * (Outputimage_width *Outputimage_height),0);


  int
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

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

void Convolution::forward_CPU(){
  float *input_pointer = thrust::raw_pointer_cast(device_FM_in.data());
  float *W_pointer = thrust::raw_pointer_cast(device_W.data());
  float *Output_point = thrust::raw_pointer_cast(device_FM_out.data());

  convLayer_forward(MiniBatch, input_pointer, Inputimage_channel, Inputimage_height,
                    Inputimage_width, W_pointer, W_width_height, Output_pointer, Outputimage_channel);
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

    int num_Thread = Inputimage_channel * Outputimage_height*Outputimage_width;
    int num_Blocks = ceil((float)num_Thread/1024);

    unroll_Kernel<<<num_blocks, 1024>>>(Inputimage_channel, Inputimage_height,Inputimage_width, W_width_height,
                                        FM_in_pointer, Unroll_FM_in_pointer);

    float *W_pointer = thust::raw_pointer_cast(device_W.data());

    dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH)
    dim3 numBlocks(ceil((float)Outputimage_width * Outputimage_height/TILE_WIDTH), ceil((float)Outputimage_channel/TILE_WIDTH));

    //void GEMM(float *W, float* Unroll_FM_in, float* FM_out, int M_height_in, int M_width_N_height_in, int N_width_in, int height_out, int width_out);
    GEMM<<<numBlocks,threadsPerBlock>>>(W_pointer, Unroll_FM_in_pointer, Output_pointer,
           Outputimage_channel,Inputimage_channel*W_width_height*W_width_height, Outputimage_width*Outputimage_height,
           Outputimage_channel, Outputimage_width*Outputimage_height);

    Output_pointer = Output_pointer+(Outputimage_channel*Outputimage_width*Outputimage_height);
    FM_in__pointer = FM_in__pointer + (Inputimage_channel*Inputimage_height*Inputimage_width);
  }
}

void Convlution::backward_GPU_gemm(){

}
}

namespace FPGA_HLS_Scope{//to do
}

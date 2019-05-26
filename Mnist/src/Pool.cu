#include "Net.h"
namespace GPU_Scope{
void Pool::init(int minib, int Inputimage_h, Inputimage_w, Inputimage_ch, int pool_size){
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,1.0);
  
  this->Inputimage_height = Inputimage_h;
  this->Inputimage_width = Inputimage_w;
  this->Inputimage_channel = Inputimage_ch;
  this->Outputimage_height = Inputimage_h/pool_size;
  this->Outputimage_width = Inputimage_w/pool_size;
  this->Outputimage_channel = Inputimage_ch;
  this->FM_out_height = minib;
  this->FM_out_weight = Outputimage_channel * Outputimage_width * Outputimage_height;
  this->MiniBatch = minib;
  this->FM_in_height = minib;
  this->FM_in_width = Inputimage_channel*Inputimage_height*Inputimage_width;
  this->Bias_height = minib;
  this->Bias_width = Inputimage_channel;
  this->pool_size = pool_size;
  
  this->host_FM_in.resize(MiniBatch*Inputimage_channel*Inputimage_height*Inputimage_width, 0);
  this->host_FM_out.resize(MiniBatch*Outputimage_channel*Outputimage_height*Outputimage_width, 0);
  this->host_Bias.resize(Inputimage_channel,0.1);
  this->device_FM_in.resize(MiniBatch*Inputimage_channel*Inputimage_height*Inputimage_width, 0);
  this->device_FM_out.resize(MiniBatch*Outputimage_channel*Outputimage_height*Outputimage_width, 0);
  this->device_Bias.resize(Inputimage_channel,0.1); 
  
}

void Pool::forward_CPU(){
	float *input_pointer = thrust::raw_pointer_cast(host_FM_in.data());
	float *Output_pointer = thrust::raw_pointer_cast(host_FM_out.data());
	poolingLayer_forward(int minib, float* FM_in, int H_in, int W_in, float* FM_out, int CH_in_out)
	
}

void Pool::forward_GPU_naive(){
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	int bz = ceil(static_cast<float>())
}


}
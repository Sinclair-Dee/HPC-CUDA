#include "Net.h"

namespace GPU_Scope{
void FullyConnect::init( int minib, int Inputimage_w_W_h, int W_w){
  //(minib, Inputimage_w_W_h) × (Inputimage_w_W_h, W_w) = (minib, W_w)
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,0.1);

  this->FM_in_height = minib;
  this->FM_in_width = Inputimage_w_W_h;
  this->FM_inT_height = Inputimage_w_W_h;
  this->FM_inT_width = minib;
  this->W_width = W_w;
  this->W_height = Inputimage_w_W_h;
  this->WT_width = Inputimage_w_W_h;
  this->WT_height = W_w;
  this->FM_out_width = W_w;
  this->Fm_out_height = minib;
  this->FM_outT_width = minib;
  this->FM_outT_width = W_w;

  this->host_FM_in.resize(FM_in_height * FM_in_width, 0);
  this->device_FM_in.resize(FM_in_height * FM_in_width, 0);
  this->device_FM_inT.resize(FM_in_width * FM_in_height, 0);
  this->host_W.resize( W_height*W_width, 0 );
  this->device_W.resize( W_width*W_height, 0 );
  for(int i = 0; i <  W_width*W_height; i++){this->device_W[i] = distribution(generator);}
  this->device_WT.resize(W_width*W_height, 0);
  this->host_Bias.resize(FM_out_width, 0)
  this->device_Bias.resize(FM_out_width, 0)
  for(int i = 0; i < FM_out_width; i++){this->device_Bias[i] = distribution(generator);}
  this->host_Wgrad.size(W_height * W_width);
  this->device_Wgrad.size(W_height * W_width);
  this->host_FM_out.size(Fm_out_height * FM_out_width);
  this->device_FM_out.size(Fm_out_height * FM_out_width);

}

void FullyConnect::forward(){
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks(ceil((float)FM_out_width)/TILE_WIDTH , ceil((float)FM_out_height/TILE_WIDTH));

  float *input_pointer = thrust::raw_pointer_cast(device_FM_in.data());
  float *W_pointer = thrust::raw_pointer_cast(device_W.data());
  float *Output_point = thrust::raw_pointer_cast(device_FM_out.data());
  float *b_pointer = thrust::raw_pointer_cast(device_Bias.data());

  GEMMwithBias<<<numBlocks,threadsPerBlock>>>(input_pointer, W_pointer, Output_point, b_pointer
                 FM_in_height, FM_in_width, W_width, Fm_out_height, FM_out_width);
}

void FullyConnect::backward()

void FullyConnect::forward_gpu_test(thrust::device_vector<float> &input, int test_number);
void FullyConnect::backward();

}

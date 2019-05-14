#include "Net.h"

void Convolution::init(int minib, int Inputimage_h, int Inputimage_w, int Inputimage_ch, int W_w_h, int W_ch){
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,0.1);
 
  this->Mini_Batch = minib;

  this->W_weight_height = W_w_h;
  this->W_channel = W_ch;//一维表达

  this->Inputimage_width = Inputimage_w;
  this->Inputimage_height = Inputimage_h;
  this->Inputimage_channel = Inputimage_ch;
  this->FM_in__width = Inputimage_ch*Inputimage_w*Inputimage_h;
  this->FM_in__height = minib;
  
  this->Outputimage_width=(Inputimage_width-W_width_height+1);
  this->Outputimage_height=(Inputimage_height-W_width_height+1);
  this->Outputimage_channel=W_channel/Inputimage_channel;
  this->FM_out_weight = Outputimage_channel*Outputimage_height*Outputimage_width; 
  this->FM_out_height = minib;

  this->Unroll_X_width = Outputimage_width*Outputimage_height;
  this->Unroll_X_height = Inputimage_channel*W_width_height*W_width_height;







}

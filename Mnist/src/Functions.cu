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

//GEMM<<<numBlocks,threadsPerBlock>>>(W_pointer, Unroll_FM_in_pointer, Output_pointer,
//       Outputimage_channel,Inputimage_channel*W_width_height*W_width_height, Outputimage_width*Outputimage_height,
//       Outputimage_channel, Outputimage_width*Outputimage_height);

__global__
void GEMM(float *W, float* Unroll_FM_in, float* FM_out, 
          int M_height_in, int M_width_N_height_in, int N_width_in,
          int height_out, int width_out){

          }



































}

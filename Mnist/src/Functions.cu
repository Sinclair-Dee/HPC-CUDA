#include "Net.h"

void clear(){
  usleep(100000);
  for(int i = 0; i < 100; i++) printf("\n")
}
namespace GPU_Scope{

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
  
  infile.read((char*)&magic_number)
}









































}

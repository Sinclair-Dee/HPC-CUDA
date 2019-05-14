#include "Net.h"

const char* PATH_TRAIN_DATA  = "../mnist/train-images-idx3-ubyte"
const char* PATH_TRAIN_LABEL = "../mnist/train-labels-idx1-ubyte"
const char* PATH_TEST_DATA   = "../mnist/t10k-images-idx3-ubyte"
const char* PATH_TEST_LABEL  = "../mnist/t10k-labels-idx1-ubyte"

/***** Main function ***********************************/
int main(){
  thrust::host_vector<thrust::host_vector<float>> DataTrain(NUM_TRAIN/MINIBATCH, thrust::host_vector<float>(RAW_PIXELS_PER_IMG_PADDING * MINIBATCH,0));
  thrust::host_vector<thrust::host_vector<float>> DataTest(NUM_TEST/MINIBATCH,   thrust::host_vector<float>(RAW_PIXELS_PER_IMG_PADDING * MINIBATCH,0));
  thrust::host_vector<int> LabelTrain(NUM_TRAIN,0);
  thrust::host_vector<int> LabelTest(MUN_TEST,0);

  //load data and label.
  GPU_Scope::read_data(PATH_TRAIN_DATA,DataTrain);
  read_label(PATH_TRAIN_LABEL,LabelTrain);
  
  read_data(PATH_TEST_DATA,DataTest);
  read_data(PATH_TEST_LABEL,LabelTest);


}

#Makefile
#define variable

objects = MaxVinArray.o kernels.o
NVCC = nvcc
opt =  -g -G
ARCH = -arch=sm_75
LIBS = 
exename = MaxVinArray

#compile
$(exename): $(objects)
	$(NVCC) $(opt) -o $(exename) $(objects) $(LIBS)
kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
MaxVinArray.o: MaxVinArray.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c MaxVinArray.cu

#clean Makefile
clean:
	rm $(objects)
#end of makefile

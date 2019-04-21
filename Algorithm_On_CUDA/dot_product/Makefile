#Makefile
#define variable

objects = DotProduct.o kernels.o
NVCC = nvcc
opt =  -g -G
ARCH = -arch=sm_75
LIBS = 
exename = DotProduct

#compile
$(exename): $(objects)
	$(NVCC) $(opt) -o $(exename) $(objects) $(LIBS)
kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
DotProduct.o: DotProduct.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c DotProduct.cu

#clean Makefile
clean:
	rm $(objects)
#end of makefile

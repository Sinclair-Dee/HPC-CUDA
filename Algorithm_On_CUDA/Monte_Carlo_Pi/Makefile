#Makefile 
#define variables
objects= Monte_Carlo_Pi.o kernels.o 
NVCC= nvcc               #cuda c compiler
opt= -O2                 #optimization flag
ARCH= -arch=sm_75        #cuda compute capability
LIBS=  
execname= Monte_Carlo_Pi


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
Monte_Carlo_Pi.o: Monte_Carlo_Pi.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c Monte_Carlo_Pi.cu


#clean Makefile
clean:
	rm $(objects)
	rm $(execname)

#end of Makefile


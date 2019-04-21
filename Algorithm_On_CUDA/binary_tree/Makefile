#Makefile 
#define variables
objects= BinaryTree.o kernels.o 
NVCC= nvcc               #cuda c compiler
opt= -O2            #optimization flag
ARCH= -arch=sm_75        #cuda compute capability
LIBS=  
execname= BinaryTree


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
BinaryTree.o: BinaryTree.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c BinaryTree.cu


#clean Makefile
clean:
	rm $(objects)
#end of Makefile


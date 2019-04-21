#Makefile 
#define variables
objects= Sieve_of_Eratosthenes.o kernels.o 
NVCC= nvcc               #cuda c compiler
opt= -O2                 #optimization flag
ARCH= -arch=sm_75        #cuda compute capability
LIBS=  
execname= Sieve_of_Eratosthenes


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
Sieve_of_Eratosthenes.o: Sieve_of_Eratosthenes.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c Sieve_of_Eratosthenes.cu


#clean Makefile
clean:
	rm $(objects)
	rm $(execname)

#end of Makefile


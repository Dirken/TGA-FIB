CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O2 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = bitonic.exe
OBJ	        = bitonic.o

default: $(EXE)

bitonic.o: multiGPU.cu
	$(NVCC) -c -o $@ multiGPU.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o *.txt S*.* $(EXE)

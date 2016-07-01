#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define NUM_THREADS     1024				 
#define NUM_BLOCKS 	32768			
#define NUM_VALUES NUM_THREADS*NUM_BLOCKS

void InitV(int *v);
void bitonic_sort(int *dev_values);
void test(int *v);

__global__ void bitonic_sort_step(int *dev_values, int j, int k){
  int i, ixj; // Sorting partners: i and ixj 
  i = threadIdx.x + blockDim.x * blockIdx.x;

  ixj = i^j;
  if ((ixj) > i) {
    if ((i & k) == 0) {
      if (dev_values[i] > dev_values[ixj]) {
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i & k) != 0) {
      if (dev_values[i] < dev_values[ixj]) {
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}




void bitonic_sort(int *dev_values){

  dim3 numBlocks(NUM_BLOCKS, 1);
  dim3 numThreads(NUM_THREADS, 1);

  int j, k;
  for (k = 2; k <= NUM_VALUES; k = 2 * k) {
    for (j = k >> 1; j > 0; j = j >> 1) {
      bitonic_sort_step<<<numBlocks, numThreads>>>(dev_values, j, k);
    }
  }
}

int main(){
  srand(time(NULL));
  int *host_values, *dev_values, *original_values;

  float TiempoTotal, TiempoKernel; 

  cudaEvent_t E0, E1, E2, E3;
  
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);  
  unsigned int numBytes = NUM_VALUES * sizeof(int);

  //obtenemos memoria en el host
  //host_values = (int*) malloc(numBytes);
  //original_values = (int*) malloc(numBytes);

  cudaMallocHost( &host_values, numBytes);
  cudaMallocHost( &original_values, numBytes);

  //inicializamos el vector 
  InitV(original_values);

  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);
  memcpy(host_values, original_values, numBytes);

  //obtenemos memoria en el device
  cudaMalloc((int**)&dev_values, numBytes);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  //Copiamos datos del host al device
  cudaMemcpy(dev_values, host_values, numBytes, cudaMemcpyHostToDevice);

  //Ejecutamos el kernel
  bitonic_sort(dev_values);
  
  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  //Obtener el resultado desde el host
  cudaMemcpy( host_values, dev_values, numBytes, cudaMemcpyDeviceToHost);

  //hacemos un test para comprobar el orden si es correcto
  test(host_values);
  
  //Liberar memoria del device y del host
  cudaFree(dev_values);
  cudaFree(original_values);
  cudaFree(host_values); 

  cudaDeviceSynchronize();
  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal, E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);

  printf("num Threads: %d\n", NUM_THREADS);
  printf("num Blocs: %d\n", NUM_BLOCKS);


  printf("Tiempo global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  cudaEventDestroy(E0); 
  cudaEventDestroy(E1); 
  cudaEventDestroy(E2); 
  cudaEventDestroy(E3);

} 

void InitV(int *v) {
   int i;
   for (i = 0; i < NUM_VALUES; i++) {
     v[i] = rand();
   }
}

void test (int *v) {
  int i;
  int val = v[0];
  for (i = 1; i < NUM_VALUES; ++i) {
    if (v[i] < val) {
	printf("val: %d, v[%d]: %d.\n", val, i, v[i]);
        printf("TEST FAIL\n\n");
	return;
    } else {
	val = v[i];
    }
  }

  printf("TEST OK\n\n");

}

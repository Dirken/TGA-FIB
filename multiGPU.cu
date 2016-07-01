#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

//#define SHARED_SIZE_LIMIT 1024
#define NUM_THREADS 1024
#define NUM_BLOCKS 32768
#define NUM_VALS NUM_THREADS*NUM_BLOCKS
#define SHARED_SIZE_LIMIT 1024


int random_float() {
  return (int)rand()/(int)RAND_MAX;
}

void array_print(int *arr, int length)  {
  int i;
  for (i = 0; i < length; ++i) {
    printf("%d ",  arr[i]);
  }
  printf("\n");
}

void array_fill(int *v) {
   int i;
   for (i = 0; i < NUM_VALS; i++) {
     v[i] = rand();
   }
}

void test (int *v) {
  int i;
  int val = v[0];
  for (i = 1; i < NUM_VALS; ++i) {
    if (val < v[i]) {
        printf("val: %d, v[%d]: %d.\n", val, i, v[i]);
        printf("TEST FAIL\n\n");
        return;
    } else {
        printf("val: %d, v[%d]: %d.\n", val, i, v[i]);

        val = v[i];
    }
  }

  printf("TEST OK\n\n");

}
   

/*
void array_fill(int *arr, int length) {
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = length-i;//random_float();
  }
}*/

void array_copy(int *dst, int *src, int length) {
  int i;
  for (i=0; i<length; ++i) {
    dst[i] = src[i];
  }
}

//Comparamos dos elementos y en caso de ser decrecientes, los swapeamos.
__device__ inline void comparator(int &A, int &B, uint dir) {
    int temp;
    if ((A <= B) == dir) {
        temp = A;
        A = B;
        B = temp;
    }
}

/*La cosa en este bitonicsort es que compartimos memoria.
Asíues pese que la idea principal es la misma, nosotros lo que hacemos es 
comparaciones entre elementos de las distintas memorias para hacer que,
finalmente el vector termine ordenado.
*/

__global__ void bitonicSortShared(int *dev_values) 
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    __shared__ int sh_values[SHARED_SIZE_LIMIT];
    sh_values[tx] = dev_values[index];
    sh_values[tx + (SHARED_SIZE_LIMIT/2)] = dev_values[index + (SHARED_SIZE_LIMIT/2)];
    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
      uint ddd = (tx & (size / 2)) == 0;//direction: ascending or descending
      for (uint stride = size/2; stride > 0; stride >>= 1) {
          __syncthreads();
          uint pos = 2 * tx - (tx & (stride - 1));
          comparator(sh_values[pos], sh_values[pos + stride], ddd);
      }
    }
    uint ddd = ((bx&1) == 0); //    uint ddd = ((bx&1)==0);
    {
      for (uint stride = SHARED_SIZE_LIMIT/2; stride > 0; stride >>= 1) {
          __syncthreads();
          uint pos = 2 * tx - (tx & (stride - 1));
          comparator(sh_values[pos + 0], sh_values[pos + stride], ddd);
      }
    }
    __syncthreads();
    dev_values[index] = sh_values[tx];
    dev_values[index+(SHARED_SIZE_LIMIT/2)] = sh_values[tx+(SHARED_SIZE_LIMIT/2)];
}

void bitonic_sort(int *values) 
{
  int *dev_values;
  size_t size = NUM_VALS * sizeof(int);
  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  dim3 numBlocks(NUM_BLOCKS, 1);
  dim3 numThreads(NUM_THREADS, 1);
  cudaDeviceSynchronize();
  uint blockCount = NUM_VALS / SHARED_SIZE_LIMIT;
  uint threadCount = SHARED_SIZE_LIMIT / 2;
  printf("blockCount=%d, threadCount=%d, SHARED_SIZE_LIMIT=%d\n", blockCount, threadCount, SHARED_SIZE_LIMIT);
  bitonicSortShared<<<blockCount, threadCount>>>(dev_values);
  cudaDeviceSynchronize();
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

int main(void) 
{
  //int *values = (int*) malloc( NUM_VALS * sizeof(int));
  //int *ref = (int*) malloc( NUM_VALS * sizeof(int));
  int *host_values;
  cudaMallocHost( &host_values, NUM_VALS * sizeof(int));
//  cudaMallocHost( &original_values, numBytes);

  float TiempoKernel; 

  cudaEvent_t E1, E2;

  cudaEventCreate(&E1);
  cudaEventCreate(&E2);

  array_fill(host_values);	 

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  cudaFuncSetCacheConfig(bitonicSortShared, cudaFuncCachePreferL1);

  bitonic_sort(host_values);
  test(host_values);
  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  cudaEventElapsedTime(&TiempoKernel, E1, E2);

  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  cudaFree(host_values);
}

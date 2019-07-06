#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda.h"

#define DATA_SIZE 1 << 28

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

template <uint blockSize>
__device__ void warpReduce(volatile uint* sharedData, int tid) {
    sharedData[tid]+=sharedData[tid+32];
    sharedData[tid]+=sharedData[tid+16];
    sharedData[tid]+=sharedData[tid+8];
    sharedData[tid]+=sharedData[tid+4];
    sharedData[tid]+=sharedData[tid+2];
    sharedData[tid]+=sharedData[tid+1];
}

template <uint blockSize>
__global__ void aggregator(uchar* globalData,ulong* sum) {
    uint x = threadIdx.x + blockIdx.x*2*blockDim.x;

    uint tid = threadIdx.x;

    if (x<DATA_SIZE) {
        __shared__ uint sharedData[1024];
        
        sharedData[tid] = globalData[x]+globalData[x+blockDim.x];

        __syncthreads();

        if (blockSize >=1024) {
            if (tid<512)
                sharedData[tid]+=sharedData[tid+512];
            __syncthreads();
        }

        if (blockSize >=512) {
            if (tid<256)
                sharedData[tid]+=sharedData[tid+256];
            __syncthreads();
        }

        if (blockSize >=256) {
            if (tid<128)
                sharedData[tid]+=sharedData[tid+128];
            __syncthreads();
        }

        if (blockSize >=128) {
            if (tid<64)
                sharedData[tid]+=sharedData[tid+64];
            __syncthreads();
        }

        if (tid<32)
            warpReduce<blockSize>(sharedData,tid);

        if (threadIdx.x==0) {
            sum[blockIdx.x]+= sharedData[0];
        }
    }
}

int main(int argc,char** argv) {
    srand(2019);
    uchar* data = (uchar*) malloc(DATA_SIZE);
    
    for (int i=0;i<DATA_SIZE;i++) {
        data[i] = rand()%256;
    }

    ulong serialCount = 0;
    double start = omp_get_wtime();
    for (uint i=0;i<DATA_SIZE;i++ ) {
        serialCount+=data[i];
    }
    double end = omp_get_wtime();
    double serialDuration = end-start;
    printf("Serial operation took %.5f seconds to run. The total is %u, Speed up -\n",serialDuration,serialCount);

    ulong parallelCount = 0; 
    start = omp_get_wtime();
    struct CudaContext cudaContext;
    cudaContext.init();
    const int numberOfThreads = 1024;
    const int numberOfBlocks = cudaContext.getBlocks(DATA_SIZE)/2;
    ulong* sums = (ulong*) malloc(sizeof(ulong)*numberOfBlocks);
    for (uint i=0;i<numberOfBlocks;i++)
        sums[i]=0;
   
    aggregator<1024><<<numberOfBlocks,numberOfThreads>>>(
       (uchar*) cudaContext.cudaIn((void*) data,DATA_SIZE),
       (ulong*) cudaContext.cudaInOut((void*) sums,sizeof(ulong)*numberOfBlocks));
    
    cudaContext.synchronize((void*)sums);

    for (int i=0;i<numberOfBlocks;i++) {
        parallelCount+=sums[i];
    }
    end = omp_get_wtime();
    double parallelDuration = end - start;
   
    printf("Parallel operation took %.5f seconds to run. The total is %u, Speed up %.2f\n",parallelDuration,parallelCount,serialDuration/parallelDuration);
    
    cudaContext.dispose();
    free(data);
    free(sums);

    printf("Finished");

    return 0;
}
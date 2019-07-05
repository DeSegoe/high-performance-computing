#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda.h"

#define DATA_SIZE 1 << 28

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

__global__ void aggregator(uchar* globalData,ulong *sum,uint N) {
    uint x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x<N && threadIdx.x<N) {
        __shared__ uint sharedData[1024];
        
        sharedData[threadIdx.x] = globalData[x];

        __syncthreads();

        int halfBlock = blockDim.x/2;

        for (int s=1;s<halfBlock;s++) {
            int index = 2*s*threadIdx.x;
            if (index<blockDim.x)
                sharedData[index]+= sharedData[index+s];

            __syncthreads();
        }

        if (threadIdx.x==0) {
            int number = 0;
            atomicAdd(&number,1);
            sum[0]+= sharedData[0];
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
    printf("Serial operation took %.5f seconds to run. The total is %u, Speed up -\n",end-start,serialCount);

    ulong parallelCount = 0;
    struct CudaContext cudaContext;
    cudaContext.init();
    start = omp_get_wtime();
    const int numberOfThreads = 1024;
    const int numberOfBlocks = cudaContext.getBlocks(DATA_SIZE);
    aggregator<<<numberOfBlocks,numberOfThreads>>>(
       (uchar*) cudaContext.cudaIn((void*) data,DATA_SIZE),
       (ulong*) cudaContext.cudaInOut((void*) &parallelCount,sizeof(ulong)),
       DATA_SIZE);
    //cudaContext.synchronize();
    end = omp_get_wtime();
    printf("Parallel operation took %.5f seconds to run. The total is %u, Speed up -\n",end-start,parallelCount);
    
    cudaContext.dispose();
    free(data);

    printf("Finished");

    return 0;
}
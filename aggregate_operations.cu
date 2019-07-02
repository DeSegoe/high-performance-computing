#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda.h"

#define DATA_SIZE 1 << 3

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

__global__ void test(int* input,int* output) {
    int index = threadIdx.x+blockIdx.x*blockDim.x +threadIdx.y+blockIdx.y*blockDim.y;
    output[index] = input[index]*2;
}

int main(int argc,char** argv) {
    srand(2019);
    int* input = (int*) malloc(sizeof(int)*DATA_SIZE);
    int* output = (int*) malloc(sizeof(int)*DATA_SIZE);
    
    for (int i=0;i<DATA_SIZE;i++) {
        input[i] = i;
        output[i] = 0;
    }

    struct CudaContext context;
    context.init();
    context.displayProperties();
    test<<<1,8>>>(
        (int*) context.cudaIn((void*) input,sizeof(uint)*DATA_SIZE),
        (int*) context.cudaInOut((void*) output,sizeof(uint)*DATA_SIZE));
    context.synchronize();
    context.dispose();

    for (int i=0;i<DATA_SIZE;i++)
        printf("%d\n",output[i]);

    free(input);
    free(output);

    printf("Finished...");

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda.h"

#define DATA_SIZE 1 << 3

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

__device__ __constant__ int constants[20];

__global__ void displayConstant() {
    int index = threadIdx.x+blockIdx.x*blockDim.x +threadIdx.y+blockIdx.y*blockDim.y;
    printf("constant: %d\n",constants[index]);
}

__global__ void test(int* input,int* output) {
    int index = threadIdx.x+blockIdx.x*blockDim.x +threadIdx.y+blockIdx.y*blockDim.y;
    output[index] = input[index]*2;
}

__global__ void countIterations(int* counter) {
    atomicAdd(counter,1);
}

int main(int argc,char** argv) {
    srand(2019);
    int* input = (int*) malloc(sizeof(int)*DATA_SIZE);
    int* input_constant = (int*)malloc(sizeof(int)*DATA_SIZE);
    int* output = (int*) malloc(sizeof(int)*DATA_SIZE);
    
    for (int i=0;i<DATA_SIZE;i++) {
        input[i] = i;
        input_constant[i] = i*i;
        output[i] = 0;
    }

    struct CudaContext context;
    context.init();
    context.displayProperties();
    context.cudaInConstant((void*) input_constant, (void**) &constants,sizeof(int)*DATA_SIZE);
    displayConstant<<<1,8>>>();
    test<<<1,8>>>(
        (int*) context.cudaIn((void*) input,sizeof(uint)*DATA_SIZE),
        (int*) context.cudaInOut((void*) output,sizeof(uint)*DATA_SIZE));
    context.synchronize();

    for (int i=0;i<DATA_SIZE;i++)
        printf("%d\n",output[i]);

    int sum = 0;
    countIterations<<<1,10>>>( (int*) context.cudaInOut((void*) &sum,sizeof(int)));
    context.synchronize();
    printf("Sum = %d\n",sum);
    context.dispose();
    free(input);
    free(input_constant);
    free(output);

    printf("Finished...");

    return 0;
}
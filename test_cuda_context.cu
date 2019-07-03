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

__global__ void updateGlobalArray(float* global) {
    int index = threadIdx.x+blockIdx.x*blockDim.x +threadIdx.y+blockIdx.y*blockDim.y;
    global[index]*=2;
}

void** generateMatrix(int elementSize,int width,int height) {
    void** result = (void**) malloc(sizeof(void*)*height);
    char** resultChar = (char**) result;
    for (int i=0;i<height;i++) {
        result[i] = (void*) malloc(elementSize*width);
        for (int j=0;j<width;j++) {
            int value = rand()%255;
            memcpy(&resultChar[i][j],&value,elementSize);
        }
    }
    return result;
}

void displayFloatMatrix(int width,int height,float** _2dArr) {
    for (int i=0;i<height;i++) {
        for (int j=0;j<width;j++) {
            printf("%.2f ", _2dArr[i][j]);
        }

        printf("\n");
    }
}

void freeMatrix(int height,void** _2dArr) {
    for (int i=0;i<height;i++)
        free(_2dArr[i]);

    free(_2dArr);
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
    //constant functionality test....
    context.cudaInConstant((void*) input_constant, (void**) &constants,sizeof(int)*DATA_SIZE);
    displayConstant<<<1,8>>>();
   
    
    //test transferring of data to and from the device
    test<<<1,8>>>(
        (int*) context.cudaIn((void*) input,sizeof(uint)*DATA_SIZE),
        (int*) context.cudaInOut((void*) output,sizeof(uint)*DATA_SIZE));
    context.synchronize();

    for (int i=0;i<DATA_SIZE;i++)
        printf("%d\n",output[i]);

    //testing shared single field functionality...
    int sum = 0;
    countIterations<<<1,10>>>( (int*) context.cudaInOut((void*) &sum,sizeof(int)));
    context.synchronize();
    printf("Sum = %d\n",sum);
    
    float** mtrx = (float**) generateMatrix(sizeof(float),2,2);
    mtrx[0][0] = 1.25;mtrx[0][1] = 1.5;
    mtrx[1][0] = 2.25;mtrx[1][1] = 2.5;
    displayFloatMatrix(2,2,mtrx);
    float* mtrxFlattened = (float*) context.cudaInOut((void**) mtrx,sizeof(float),2,2);
    updateGlobalArray<<<1,4>>>(mtrxFlattened);
    context.synchronize();
    printf("After updating\n");
    displayFloatMatrix(2,2,mtrx);
    context.dispose();

    free(input);
    free(input_constant);
    free(output);
    freeMatrix(3,(void**)mtrx);

    printf("Finished...");

    return 0;
}
#ifndef CUDA_HELPER
#define CUDA_HELPER
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char uchar;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define MAX_ARRAYS 100

struct CudaContext {
    uint* sizes;
    void** cudaPointers;
    void** hostPointers;
    int cudaPointerCount = 0;

    int deviceCount;

    void init() {
        HANDLE_ERROR(cuInit(0));
        HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
        cudaPointers = (void**) malloc(sizeof(int*)*MAX_ARRAYS);
        hostPointers = (void**) malloc(sizeof(int*)*MAX_ARRAYS);
        sizes = (uint*) malloc(sizeof(uint)*MAX_ARRAYS);
    }

    int* cudaInInt(int* hostInput,uint sizeInBytes) {
        return (int*)  cudaIn((void**) hostInput,sizeInBytes);
    }

    void** cudaIn(void** hostInput,uint sizeInBytes) {
        void** deviceInput;
        HANDLE_ERROR(cudaMalloc(deviceInput,sizeInBytes));
        HANDLE_ERROR(cudaMemcpy(deviceInput,hostInput,sizeInBytes,cudaMemcpyDeviceToHost));
        cudaPointers[cudaPointerCount] = deviceInput;
        hostPointers[cudaPointerCount] = hostInput;
        sizes[cudaPointerCount] = sizeInBytes;
        cudaPointer++;
        return deviceInput;
    }

     void** cudaInOut(void** hostInput,uint sizeInBytes) {
        int previousCudaPointer = cudaPointerCount;
        void** response = cudaIn(hostInput,sizeInBytes);
        hostPointers[previousCudaPointer] = NULL;
        return response;
    }

    void displayProperties() {
        for (int i=0;i<deviceCount;i++) {
            cudaDeviceProp prop;
            HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
            printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            printf("  Memory Clock Rate (KHz): %d\n",
                prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        }
    }

    void dispose() {
        free(sizes);
        for (int i=0;i<cudaPointerCount;i++) {
            cudaFree(cudaPointers[i]);
        }
    }
} CudaContext;

#endif

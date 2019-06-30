#ifndef CUDA_HELPER
#define CUDA_HELPER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    uchar* isOutput;
    void** devicePointers;
    void** hostPointers;
    
    int cudaPointerCount = 0;
    int deviceCount;

    void init() {
        HANDLE_ERROR(cuInit(0));
        HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
        devicePointers = (void**) malloc(sizeof(int*)*MAX_ARRAYS);
        hostPointers = (void**) malloc(sizeof(int*)*MAX_ARRAYS);
        sizes = (uint*) malloc(sizeof(uint)*MAX_ARRAYS);
        isOutput = (uchar*) malloc(sizeof(uchar)*MAX_ARRAYS);
        memset(isOutput,0,sizeof(uchar)*MAX_ARRAYS);
    }

    void* cudaIn(void* hostData,uint sizeInBytes) {
        void* deviceData;
        HANDLE_ERROR(cudaMalloc((void**)&deviceData,sizeInBytes));
        HANDLE_ERROR(cudaMemcpy(deviceData,hostData,sizeInBytes,cudaMemcpyDeviceToHost));
        devicePointers[cudaPointerCount] = deviceData;
        hostPointers[cudaPointerCount] = hostData;
        sizes[cudaPointerCount] = sizeInBytes;
        cudaPointer++;
        return deviceData;
    }

     void* cudaInOut(void* hostData,uint sizeInBytes) {
        isOutput[cudaPointerCount] = 1;
        return cudaIn(hostData,sizeInBytes);
    }

    void synchronize() {
        for (int i=0;i<cudaPointerCount;i++) {
            if (isOutput[i]) {
                HANDLE_ERROR(cudaMemcpy(hostPointers[i],devicePointers[i],sizes[i],cudaMemcpyDeviceToHost));
            }
        }

        for (int i=0;i<cudaPointerCount;i++) {
            if (!isOutput[i]) {
                HANDLE_ERROR(cudaMemcpy(devicePointers[i],hostPointers[i],sizes[i],cudaMemcpyHostToDevice));
            }
        }
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
        free(isOutput);
        for (int i=0;i<cudaPointerCount;i++) {
            cudaFree(cudaPointers[i]);
        }
    }
} CudaContext;

#endif

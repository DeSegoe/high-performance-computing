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

struct Dimensions {
    int height;
    int width;
    int sizeofElement;
    size_t pitch;
    texture<void,2,cudaReadModeElementType> tex_w;
};

struct CudaContext {
    uint* sizes;
    uchar* isOutput;
    uchar* isConstant;
    uchar* createdByContext;
    struct Dimensions* dimensions;
    void** devicePointers;
    void** hostPointers;
    void*** twoDimensionalHostPointers;
    
    int cudaPointerCount = 0;
    int deviceCount;

    void init() {
        //HANDLE_ERROR(cuInit(0));
        HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
        devicePointers = (void**) malloc(sizeof(void*)*MAX_ARRAYS);
        hostPointers = (void**) malloc(sizeof(void*)*MAX_ARRAYS);
        twoDimensionalHostPointers = (void***) malloc(sizeof(void**)*MAX_ARRAYS);
        sizes = (uint*) malloc(sizeof(uint)*MAX_ARRAYS);
        isOutput = (uchar*) malloc(sizeof(uchar)*MAX_ARRAYS);
        isConstant = (uchar*) malloc(sizeof(uchar)*MAX_ARRAYS);
        createdByContext = (uchar*) malloc(sizeof(uchar)*MAX_ARRAYS);
        dimensions = (struct Dimensions*) malloc(sizeof(struct Dimensions)*MAX_ARRAYS);
        memset(isOutput,0,sizeof(uchar)*MAX_ARRAYS);
        memset(createdByContext,0,sizeof(uchar)*MAX_ARRAYS);
        memset(isConstant,0,sizeof(uchar)*MAX_ARRAYS);
        memset(sizes,0,sizeof(uint)*MAX_ARRAYS);

        for (int i=0;i<MAX_ARRAYS;i++) {
            struct Dimensions newDimensions;
            newDimensions.height = 0;
            newDimensions.width = 0;
            newDimensions.sizeofElement = 0;
            newDimensions.pitch = 0;
            newDimensions.tex_w = NULL;
            dimensions[i] = newDimensions;
        }
    }

    void cudaInConstant(void* hostData, void* deviceData,uint sizeInBytes) {
        HANDLE_ERROR(cudaMemcpyToSymbol(hostData,deviceData,sizeInBytes));
        devicePointers[cudaPointerCount] = deviceData;
        hostPointers[cudaPointerCount] = hostData;
        isConstant[cudaPointerCount] = 1;
        sizes[cudaPointerCount] = sizeInBytes;
        cudaPointerCount++;
    }

    void cudaInTexture(texture<void,2,cudaReadModeElementType> tex_w,void** hostData,int width,int height,int sizeOfElement) {
        void* tex_arr;
        size_t pitch;
        uint widthSize = width*sizeOfElement;
        HANDLE_ERROR( cudaMallocPitch((void**)&tex_arr, &pitch, widthSize, height) );
        HANDLE_ERROR( cudaMemcpy2D(tex_arr,             // device destination                                   
                            pitch,           // device pitch (calculated above)                      
                            hostData,               // src on host                                          
                            widthSize, // pitch on src (no padding so just width of row)       
                            widthSize, // width of data in bytes                               
                            height,            // height of data                                       
                            cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaBindTexture2D(NULL, tex_w, tex_arr, tex_w.channelDesc, width, height, pitch) );
        twoDimensionalHostPointers[cudaPointerCount] = hostData;
        devicePointers[cudaPointerCount] = tex_arr;

        struct Dimensions currentDimension = dimensions[cudaPointerCount];
        currentDimension.height = height;
        currentDimension.width = width;
        currentDimension.sizeofElement = sizeOfElement;
        currentDimension.pitch = pitch;
        currentDimension.tex_w = tex_w;
        dimensions[cudaPointerCount] = currentDimension;

        cudaPointerCount++;
    }

    void* cudaIn(void* hostData,uint sizeInBytes) {
        void* deviceData;
        HANDLE_ERROR(cudaMalloc((void**)&deviceData,sizeInBytes));
        HANDLE_ERROR(cudaMemcpy(deviceData,hostData,sizeInBytes,cudaMemcpyHostToDevice));
        devicePointers[cudaPointerCount] = deviceData;
        hostPointers[cudaPointerCount] = hostData;
        sizes[cudaPointerCount] = sizeInBytes;
        cudaPointerCount++;
        return deviceData;
    }

    void* cudaInOut(void* hostData,uint sizeInBytes) {
        isOutput[cudaPointerCount] = 1;
        return cudaIn(hostData,sizeInBytes);
    }

    void* cudaIn(void** hostData,int elementSize,int width,int height) {
        uint size = width*height*elementSize;
        uint widthSize = width*elementSize;
        
        char** hostDataAsChar = (char**) hostData;
        void* hostDataFlattened = (void*) malloc(size);
        char* hostDataFlattenedAsChar = (char*) hostDataFlattened;

        for (int y=0;y<height;y++) {
            char* rowData = hostDataAsChar[y];
            memcpy(&hostDataFlattenedAsChar[y*widthSize],rowData,widthSize);
        }

        struct Dimensions currentDimension = dimensions[cudaPointerCount];
        currentDimension.height = height;
        currentDimension.width = width;
        currentDimension.sizeofElement = elementSize;
        dimensions[cudaPointerCount] = currentDimension;

        twoDimensionalHostPointers[cudaPointerCount] = hostData;

        createdByContext[cudaPointerCount] = 1;

        return cudaIn(hostDataFlattened,size);
    }

    void* cudaInOut(void** hostData,int elementSize,int width,int height) {
        isOutput[cudaPointerCount] = 1;
        return cudaIn(hostData,elementSize,width,height);
    }

    void synchronize() {
        for (int i=0;i<cudaPointerCount;i++) {
            if (isOutput[i]) {
                HANDLE_ERROR(cudaMemcpy(hostPointers[i],devicePointers[i],sizes[i],cudaMemcpyDeviceToHost));
                struct Dimensions currentDimension = dimensions[i];
                if (currentDimension.width>0 && currentDimension.height>0 && currentDimension.sizeofElement>0) {
                    char** mtx = (char**) twoDimensionalHostPointers[i];
                    char* hostPointerAsChar = (char*) hostPointers[i];
                    int rowSize = currentDimension.width*currentDimension.sizeofElement;

                    for (int i=0;i<currentDimension.height;i++) {
                        char* row = mtx[i];
                        memcpy(&hostPointerAsChar[i*rowSize],row,rowSize);  
                    }
                }
            }
        }

        for (int i=0;i<cudaPointerCount;i++) {
            if (!isOutput[i]) {
                struct Dimensions currentDimension = dimensions[i];
                if (!isConstant[i]) {
                    if (currentDimension.pitch==0)
                        HANDLE_ERROR(cudaMemcpy(devicePointers[i],hostPointers[i],sizes[i],cudaMemcpyHostToDevice));
                    else
                         HANDLE_ERROR( cudaMemcpy2D(devicePointers[i],             // device destination                                   
                            currentDimension.pitch,           // device pitch (calculated above)                      
                            twoDimensionalHostPointers[i],               // src on host                                          
                            currentDimension.sizeofElement*currentDimension.width, // pitch on src (no padding so just width of row)       
                            currentDimension.sizeofElement*currentDimension.width, // width of data in bytes                               
                            currentDimension.height,            // height of data                                       
                            cudaMemcpyHostToDevice) );
                }
                else
                    HANDLE_ERROR(cudaMemcpyToSymbol(hostPointers[i],devicePointers[i],sizes[i]));
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
        for (int i=0;i<cudaPointerCount;i++) {
            cudaFree(devicePointers[i]);
            if (createdByContext[i]) {
                free(hostPointers[i]);//we added this to memory
            }

            struct Dimensions currentDimension = dimensions[i];
            if (currentDimension.pitch>0) {
                 cudaUnbindTexture(currentDimension.tex_w);
            }
        }

        free(createdByContext);
        free(sizes);
        free(isOutput);
        free(isConstant);
        free(dimensions);
        free(hostPointers);
        free(twoDimensionalHostPointers);
        free(devicePointers);
    }
};

#endif

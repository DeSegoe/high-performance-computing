#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char uchar;
typedef unsigned int uint;

void init(void** arg,int size) {
    *arg = malloc(size);
}

void* copy(void* src,int size) {
    void* newArr = malloc(size);
    memcpy(newArr,src,size);
    return newArr;
}

void* flatten(void** hostData,int elementSize,int width,int height) {
    uint size = width*height*elementSize;
    uint widthSize = width*elementSize;
    
    char** hostDataAsChar = (char**) hostData;
    void* hostDataFlattened = (void*) malloc(size);
    char* hostDataFlattenedAsChar = (char*) hostDataFlattened;

    for (int y=0;y<height;y++) {
        char* rowData = hostDataAsChar[y];
        memcpy(&hostDataFlattenedAsChar[y*widthSize],rowData,widthSize);
    }

    return hostDataFlattened;
}

void** generateMatrix(int elementSize,int width,int height) {
    void** result = (void**) malloc(sizeof(void*)*height);
    char** resultChar = (char**) result;
    for (int i=0;i<height;i++) {
        int offset = height*width;
        result[i] = (void*) malloc(elementSize*width);
        for (int j=0;j<width;j++) {
            int value = rand()%255;
            memcpy(&resultChar[i][j],&value,elementSize);
        }
    }
    return result;
}

void displayMatrix(int** arr,int width,int height) {
    for (int i=0;i<height;i++) {
        for (int j=0;j<width;j++) {
            printf("%d ",arr[i][j]);
        }
        printf("\n");
    }
}

void displayFloatMatrix(float** arr,int width,int height) {
    for (int i=0;i<height;i++) {
        for (int j=0;j<width;j++) {
            printf("%.2f ",arr[i][j]);
        }
        printf("\n");
    }
}

void freeMatrix(void** mtx,int height) {
    for (int i=0;i<height;i++)
        free(mtx[i]);
}

int main(int argc,char** argv) {
    int* arr;
    init((void**) &arr,sizeof(int)*10);

    for (int i=0;i<10;i++) {
        arr[i] = i;
    }

    uchar* unsignedArr = (uchar*) malloc(sizeof(uchar)*10);
    memset(unsignedArr,1,sizeof(uchar)*10);

    int* cpyArr = (int*) copy((void*) arr,sizeof(int)*10);

    for (int i=0;i<10;i++) {
        printf("%d & %u\n",cpyArr[i],unsignedArr[i]);
    }

    free(arr);
    free(cpyArr);

    //other pointer magic
    float** mtx = (float**) generateMatrix(sizeof(float),3,4);
    mtx[1][1] = 67;
    float* mtxFlattened = (float*) flatten((void**)mtx,sizeof(float),3,4);
    displayFloatMatrix(mtx,3,4);
    freeMatrix((void**) mtx,4);

    printf("\n");
    for (int i=0;i<12;i++)
        printf("%d. %.2f\n",i,mtxFlattened[i]);
    free(mtxFlattened);
    printf("Finished.");
    return 1;
}
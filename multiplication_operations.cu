#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cuda.h"

#define BLOCK_SIZE 32

__global__ void multiplyVectors(float* A, float* B, float*C,int WIDTH,int HEIGHT) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if (x<WIDTH && y<HEIGHT) {

        double result = 0.0;

        for (int i=0;i<WIDTH;i++)
            result+=A[y*WIDTH+i]*B[i*WIDTH+x];

        C[y*WIDTH+x] = result;
    }
}

float** generateTwoDimensionalArray(int width,int height, int max) {
    srand(2019);
    float** mtx = (float**) malloc(sizeof(float*)*height);
    
    for (int i=0;i<height;i++) {
        float* row = (float*) malloc(sizeof(float)*width);
        for (int j=0;j<width;j++) {
            row[j] = (float) (rand()%max);
        }
        mtx[i] = row;
    }

    return mtx;
}

float* generateRow(int width,int max) {
    srand(2019);
    float* row = (float*) malloc(sizeof(float)*width);
    for (int i=0;i<width;i++)
        row[i] = (float) (rand()%max);
    return row;
}

void display(float** mtx,int width,int height) {
    for (int i=0;i<height;i++) {
        for (int j=0;j<width;j++) {
            printf("%.2f ",mtx[i][j]);
        }
        printf("\n");
    }
}

void validate(float** mtx1,float** mtx2,int width,int height) {
    int numErrors = 0;

    for (int i=0;i<height;i++) {
        for (int j=0;j<width;j++) {
            if (mtx1[i][j]!=mtx2[i][j])
                numErrors++;
        }
    }

    if (numErrors>0)
        printf("Found %d errors\n",numErrors);
    else
        printf("Passed validation\n");
}

void dispose(float** mtx,int height) {
    for (int i=0;i<height;i++) {
        free(mtx[i]);
    }

    free(mtx);
}

int main(int argc, char**argv) {
    const int WIDTH = 1500;
    const int HEIGHT = 1500;

    float** A = generateTwoDimensionalArray(WIDTH,HEIGHT,1000);
    float** B = generateTwoDimensionalArray(WIDTH,HEIGHT,1000);
    float** CSerial = generateTwoDimensionalArray(WIDTH,HEIGHT,1);
    float** CParallel = generateTwoDimensionalArray(WIDTH,HEIGHT,1);
    float** CCuda = generateTwoDimensionalArray(WIDTH,HEIGHT,1);

    double serialDuration = 0,parallelDuration=0,cudaDuration=0;
    double start = 0,end = 0;

    start = omp_get_wtime();

    for (int i=0;i<HEIGHT;i++) {
        for (int j=0;j<WIDTH;j++) {
            double result = 0;
            for (int k=0;k<HEIGHT;k++) {
                result+=A[i][k]*B[k][j];
            }
            CSerial[i][j] = result;
        }
    }

    end = omp_get_wtime();
    serialDuration = end-start;

    printf("The %dx%d serial matrix multiplication took %.5f seconds to run. Speed up -\n",WIDTH,HEIGHT,serialDuration);

    start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int i=0;i<HEIGHT;i++) {
        for (int j=0;j<WIDTH;j++) {
            double result = 0;
            for (int k=0;k<HEIGHT;k++) {
                result+=A[i][k]*B[k][j];
            }
            CParallel[i][j] = result;
        }
    }

    end = omp_get_wtime();
    parallelDuration = end-start;

    printf("The %dx%d parallel matrix multiplication took %.5f seconds to run. Speed up %.2f\n",WIDTH,HEIGHT,parallelDuration,serialDuration/parallelDuration);

    validate(CSerial,CParallel,WIDTH,HEIGHT);

    start = omp_get_wtime();
    struct CudaContext cudaContext;
    cudaContext.init();

    dim3 gridDim = cudaContext.getBlocks(WIDTH,HEIGHT,BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);

    multiplyVectors<<<gridDim,blockDim>>>(
        (float*) cudaContext.cudaIn((void**) A,sizeof(float),WIDTH,HEIGHT),
        (float*) cudaContext.cudaIn((void**) B,sizeof(float),WIDTH,HEIGHT),
        (float*) cudaContext.cudaInOut((void**) CCuda,sizeof(float),WIDTH,HEIGHT),
        WIDTH,
        HEIGHT);
    cudaContext.synchronize((void**) CCuda,WIDTH,HEIGHT);
    end = omp_get_wtime();
    cudaDuration = end-start;

    printf("The %dx%d cuda matrix multiplication took %.5f seconds to run. Speed up %.2f\n",WIDTH,HEIGHT,cudaDuration,serialDuration/cudaDuration);

    validate(CSerial,CCuda,WIDTH,HEIGHT);
    cudaContext.dispose();

    dispose(A,HEIGHT);
    dispose(B,HEIGHT);
    dispose(CSerial,HEIGHT);
    dispose(CParallel,HEIGHT);
    dispose(CCuda,HEIGHT);

    return 0;
}
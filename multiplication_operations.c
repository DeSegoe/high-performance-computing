#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
        printf("Found %d errors\n");
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
    const int WIDTH = 1000;
    const int HEIGHT = 1000;
    const int N = WIDTH*HEIGHT;
    const int ROW_SIZE = sizeof(float)*WIDTH;
    const int SIZE = sizeof(float)*N; 

    float** A = generateTwoDimensionalArray(WIDTH,HEIGHT,1000);
    float** B = generateTwoDimensionalArray(WIDTH,HEIGHT,1000);
    float** CSerial = generateTwoDimensionalArray(WIDTH,HEIGHT,1);
    float** CParallel = generateTwoDimensionalArray(WIDTH,HEIGHT,1);

    double serialDuration = 0,parallelDuration=0;
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

    dispose(A,HEIGHT);
    dispose(B,HEIGHT);
    dispose(CSerial,HEIGHT);
    dispose(CParallel,HEIGHT);

    return 0;
}
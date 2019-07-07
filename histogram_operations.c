#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

void validate(ulong arr1[256],ulong arr2[256]) {
    uchar incorrectCount = 0;
    for (int i=0;i<256;i++) {
        if (arr1[i]!=arr2[i])
            incorrectCount++;
    }

    if (incorrectCount==0)
        printf("Passed validation\n");
    else
        printf("%u bins were incorrect\n",incorrectCount);
}

int main(int argc,char** argv) {
    ulong DATA_SIZE  = 1 << 28;
    srand(2019);
    uchar* data = (uchar*) malloc(DATA_SIZE);
    
    for (uint i=0;i<DATA_SIZE;i++) {
        data[i] = rand()%256;
    }

    double serialDuration = -1;
    double start = omp_get_wtime();
    ulong serialHistogram[256];
    memset(serialHistogram,0,sizeof(ulong)*256);
    for (uint i=0;i<DATA_SIZE;i++ ) {
        serialHistogram[data[i]]++;
    }
    double end = omp_get_wtime();
    serialDuration = end - start;
    printf("Serial operation took %.5f seconds to run.Speed up -\n",serialDuration);

    double parallelDuration = -1;
    start = omp_get_wtime();
    ulong parallelHistogram[256];
    memset(parallelHistogram,0,sizeof(ulong)*256);

    #pragma omp parallel
    {
        ulong partialHistogram[256];
        memset(partialHistogram,0,sizeof(ulong)*256);
        int numThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        ulong blockSize = DATA_SIZE/numThreads;
        ulong startIndex = blockSize*tid;
        ulong endIndex = startIndex+blockSize;

        if (tid == numThreads-1)
            endIndex+= DATA_SIZE%numThreads;
        
        for (ulong i=startIndex;i<endIndex;i++) {
            partialHistogram[data[i]]++;
        }

        for (int i=0;i<256;i++) {
            #pragma omp critical
            {
                parallelHistogram[i]+=partialHistogram[i];
            }
        }
    }

    end = omp_get_wtime();
    parallelDuration = end-start;
    printf("Parallel operation took %.5f seconds to run. Speed up %.1f\n",parallelDuration,serialDuration/parallelDuration);

    validate(serialHistogram,parallelHistogram);
    free(data);

    return 0;
}
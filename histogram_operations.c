#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DATA_SIZE 1 << 28

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

int main(int argc,char** argv) {
    srand(2019);
    uchar* data = (uchar*) malloc(DATA_SIZE);
    
    for (int i=0;i<DATA_SIZE;i++) {
        data[i] = rand()%256;
    }

    ulong parallelCount = 0;
    ulong serialCount = 0;
    double serialDuration = -1;
    double start = omp_get_wtime();
    for (uint i=0;i<DATA_SIZE;i++ ) {
        serialCount+=data[i];
    }
    double end = omp_get_wtime();
    serialDuration = end - start;
    printf("Serial operation took %.5f seconds to run. The total is %u, Speed up -\n",end-start,serialCount);

    parallelCount = 0;
    start = omp_get_wtime();
    ulong partialArray[32];
    #pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int maxThreads = omp_get_num_threads();
        partialArray[idx] = 0;

        #pragma omp block

        #pragma omp for
        for (uint i=0;i<DATA_SIZE;i++) {
            partialArray[omp_get_thread_num()]+=data[i];
        }


        #pragma omp master
        {
            for (int i=0;i<maxThreads;i++) {
                parallelCount+=partialArray[i];
            }
        }
    }
    end = omp_get_wtime();
    printf("Parallel operation using partialSum[32] took %.5f seconds to run. The total is %u Speed up %.1f\n",end-start,parallelCount,serialDuration/(end-start));

    parallelCount = 0;
    start = omp_get_wtime();
    ulong partialMatrix[32][1];
    #pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int maxThreads = omp_get_num_threads();
        partialMatrix[idx][0] = 0;

        #pragma omp block

        #pragma omp for
        for (uint i=0;i<DATA_SIZE;i++) {
            partialMatrix[omp_get_thread_num()][0]+=data[i];
        }


        #pragma omp master
        {
            for (int i=0;i<maxThreads;i++) {
                parallelCount+=partialMatrix[i][0];
            }
        }
    }
    end = omp_get_wtime();
    printf("Parallel operation using partialSum[32][1] took %.5f seconds to run. The total is %u Speed up %.1f\n",end-start,parallelCount,serialDuration/(end-start));

    parallelCount = 0;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        int maxThreads = omp_get_num_threads();

        ulong threadSum = 0;

        #pragma omp for
        for (uint i=0;i<DATA_SIZE;i++) {
            threadSum+=data[i];
        }

        #pragma omp atomic
        parallelCount+=threadSum;
    }
    end = omp_get_wtime();
    printf("Parallel operation using threadSum took %.5f seconds to run. The total is %u Speed up %.1f\n",end-start,parallelCount,serialDuration/(end-start));

    parallelCount = 0;
    start = omp_get_wtime();
    parallelCount=0;
    #pragma omp parallel for reduction(+:parallelCount)
    for (uint i=0;i<DATA_SIZE;i++) {
        parallelCount+=data[i];
    }
    end = omp_get_wtime();
    printf("Parallel operation using reduction took %.5f seconds to run. The total is %u Speed up %.1f\n",end-start,parallelCount,serialDuration/(end-start));

    free(data);

    return 0;
}
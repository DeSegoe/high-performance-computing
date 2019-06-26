#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DATA_SIZE 1 << 26

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
    ulong start = omp_get_wtime();
    for (uint i=0;i<DATA_SIZE;i++ ) {
        serialCount+=data[i];
    }
    ulong end = omp_get_wtime();
    printf("Serial operation took %u to run. The total is %u\n",end-start,serialCount);

    start = omp_get_wtime();
    ulong partialSum[32][1];
    #pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int maxThreads = omp_get_num_threads();
        partialSum[idx][0] = 0;

        #pragma omp block

        #pragma omp for
        for (uint i=0;i<DATA_SIZE;i++) {
            partialSum[omp_get_thread_num()][0]+=data[i];
        }


        #pragma omp master
        {
            for (int i=0;i<maxThreads;i++) {
                parallelCount+=partialSum[i][0];
            }
        }
    }
    end = omp_get_wtime();
    printf("Parallel operation took %u to run. The total is %u\n",end-start,parallelCount);

    free(data);

    return 0;
}
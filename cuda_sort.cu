#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda.h"

#define N 1 << 28
#define DATA_SIZE sizeof(int)*N


void serialSort(int* data,int* copy, const int begin,const int end) {
    if (begin<end) {
        int middle = (end+begin)/2;
        serialSort(data,copy,begin,middle);
        serialSort(data,copy,middle+1,end);

        for (int i=begin;i<=end;i++)
            copy[i] = data[i];

        int i=begin;
        int j = middle+1;
        int index = begin;
        while (1) {
            if (i<=middle && (j>end || copy[i]<=copy[j]))
                data[index++] = copy[i++];
            else if (j<=end && (i>middle || copy[j]<copy[i]))
                data[index++] = copy[j++];
            else
                break;
        }
    }
}

void serialSort(int* data) {
    int* copy = (int*) malloc(DATA_SIZE);
    for (int i=0;i<N;i++)
        copy[i] = data[i];

    int size  = (int) N;
    serialSort(data,copy,0,size-1);

    free(copy);
}

void validate(char* method,int*data) {
    int previous = data[0];
    for (int i=1;i<N;i++) {
        if (previous>data[i]) {
            printf("The sort is invalid. %d found at %d, is greater than %d found at %d\n",previous,i-1,data[i],i);
            return;
        }
        previous = data[i];
    }

    printf("%s implementation is valid\n",method);
}

int main(int argc,char** argv) {
    int* data = (int*) malloc(DATA_SIZE);
    
    srand(2019);
    for (int i=0;i<N;i++)
        *(data+i) = rand()%30000;
    
    double start = omp_get_wtime();
    serialSort(data);
    double end = omp_get_wtime();
    double serialDuration = end-start;
    printf("Serial merge sort took %.5f seconds to run. Speed up -\n",serialDuration);
    validate("Serial",data);
    free(data);
    return 1;
}
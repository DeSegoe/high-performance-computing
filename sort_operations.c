#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1 << 27
#define DATA_SIZE sizeof(int)*N


void serialMergeSort(int* data,int* copy, const int begin,const int end) {
    if (begin<end) {
        int middle = (end+begin)/2;
        serialMergeSort(data,copy,begin,middle);
        serialMergeSort(data,copy,middle+1,end);

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

void parallelMergeSort(int* data,int* copy, const int begin,const int end,const int num_threads,const int level) {
    //printf("Thread number: %d. Num threads:%d && %d\n",omp_get_thread_num(),num_threads, (int)pow(2,level));
    if (begin<end) {
        int middle = (end+begin)/2;
        if(num_threads-1> (int)(pow(2,level))) {
            #pragma omp taskgroup
            {
                #pragma omp task
                parallelMergeSort(data,copy,begin,middle,num_threads,level+1);
                #pragma omp task
                parallelMergeSort(data,copy,middle+1,end,num_threads,level+1);
            }
        }
        else {
            serialMergeSort(data,copy,begin,middle);
            serialMergeSort(data,copy,middle+1,end);
        }
        
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
    serialMergeSort(data,copy,0,size-1);

    free(copy);
}

void parallelSort(int* data) {
    int* copy = (int*) malloc(DATA_SIZE);
    for (int i=0;i<N;i++)
        copy[i] = data[i];

    int size  = (int) N;

    #pragma omp parallel
    {
        #pragma omp master
        {
            parallelMergeSort(data,copy,0,size-1,omp_get_num_threads(),0);
        }
    }
    
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

    for (int i=0;i<N;i++)
        *(data+i) = rand()%30000;

    start = omp_get_wtime();
    parallelSort(data);
    end = omp_get_wtime();
    double parallelDuration = end-start;
    printf("Parallel merge sort took %.5f seconds to run. Speed up %.2f\n",parallelDuration,serialDuration/parallelDuration);
    validate("Parallel",data);
    
    free(data);
    return 1;
}
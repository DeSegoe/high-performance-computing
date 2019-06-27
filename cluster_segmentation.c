#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "bmp_image_util.h"

struct Centroid {
    float r;
    float g;
    float b;
    ulong count;
};

void validate(int argc,char** argv) {
    if (argc!=2) {
        printf("Invalid command. To run the app %s <filename>.bmp\n",argv[0]);
        exit(-1);
    }
}

struct Centroid createCentroid(struct BmpImage bmpImage,int index) {
    struct Centroid newCentroid;
    newCentroid.r = (float) bmpImage.r_channel[index];
    newCentroid.g = (float) bmpImage.g_channel[index];
    newCentroid.b = (float) bmpImage.b_channel[index];
    return newCentroid;
}

struct BmpImage performClusterSegmentation(struct BmpImage bmpImage) {
    int NUMBER_OF_CLUSTERS;
    printf("Please enter the number of centroids:");
    scanf("%d",&NUMBER_OF_CLUSTERS);
    double start = omp_get_wtime();
    const uint size = bmpImage.image_height*bmpImage.image_width;
    uchar* clusters_indexes = (uchar*) malloc(size);
    struct Centroid* centroids = (struct Centroid*) malloc(sizeof(struct Centroid)*NUMBER_OF_CLUSTERS);
    struct Centroid* aggregates = (struct Centroid*) malloc(sizeof(struct Centroid)*NUMBER_OF_CLUSTERS);
    srand(2019);

    for (int i=0;i<NUMBER_OF_CLUSTERS;i++)
        centroids[i] = createCentroid(bmpImage, rand()%size);

    for (int i=0;i<size;i++)
        clusters_indexes[i] = rand()%NUMBER_OF_CLUSTERS;

    const uchar MAX_ITERATIONS = 50;

    int index = 0;

    while (index<MAX_ITERATIONS) {
        if (index%10==0)
            printf("%.2f complete...\n",100*((float) index)/MAX_ITERATIONS);
        index++;
        for (int i=0;i<NUMBER_OF_CLUSTERS;i++) {
            struct Centroid newCentroid;
            newCentroid.r = 0;
            newCentroid.g = 0;
            newCentroid.b = 0;
            newCentroid.count = 0;
            aggregates[i] = newCentroid;
        }
        #pragma omp parallel for
        for (int i=0;i<size;i++) {
            uchar group_index = clusters_indexes[i];
            struct Centroid ithPixel = createCentroid(bmpImage,i);
            float minEuclidean = 1e6;
            uchar selectedIndex = group_index;
            for (int j=0;j<NUMBER_OF_CLUSTERS;j++) {
                struct Centroid currentCentroid = centroids[j];
                float distance =    pow((pow(ithPixel.r -currentCentroid.r,2)+
                                    pow(ithPixel.g -currentCentroid.g,2)+
                                    pow(ithPixel.b - currentCentroid.b,2)),0.5);
                
                if (distance<minEuclidean) {
                    minEuclidean = distance;
                    selectedIndex = j;
                }
            }

             if (selectedIndex!=group_index) {
                clusters_indexes[i] = selectedIndex;
                ithPixel = createCentroid(bmpImage,selectedIndex);
            }
        }

        for (int i=0;i<size;i++) {
            struct Centroid ithPixel = createCentroid(bmpImage,i);
            struct Centroid allocatedCentroid = aggregates[clusters_indexes[i]];
            allocatedCentroid.r+=ithPixel.r;
            allocatedCentroid.g+=ithPixel.g;
            allocatedCentroid.b+=ithPixel.b;
            allocatedCentroid.count++;
        }

        for (int i=0;i<NUMBER_OF_CLUSTERS;i++) {
            struct Centroid centroid = centroids[i];
            struct Centroid aggregate = aggregates[i];
            centroid.r = aggregate.r/aggregate.count;
            centroid.g = aggregate.g/aggregate.count;
            centroid.b = aggregate.b/aggregate.count;
        }
    }

    #pragma omp parallel for
    for (int i=0;i<size;i++) {
         struct Centroid centroid = centroids[clusters_indexes[i]];
         bmpImage.r_channel[i] = (uchar) centroid.r;
         bmpImage.g_channel[i] = (uchar) centroid.g;
         bmpImage.b_channel[i] = (uchar) centroid.b;
    }

    for (int i=0;i<NUMBER_OF_CLUSTERS;i++) {
        struct Centroid centroid = centroids[i];
        printf("%d (%u,%u,%u)\n",i,(uchar) centroid.r,(uchar) centroid.g,(uchar) centroid.b);
    }
   
   free(centroids);
   free(aggregates);
   free(clusters_indexes);
   double end = omp_get_wtime();
   printf("Duration %.6f\n",end-start);
   return bmpImage;
}

int main(int argc, char** argv) {
    validate(argc,argv);
    char* filePath = argv[1];
    struct BmpImage bmpImage = loadWindowsBpm(filePath);
    displayBpmImageMetaData(bmpImage);
    bmpImage = performClusterSegmentation(bmpImage);
    writeBpmImage("./data/cluster_output.bmp",bmpImage);
    disposeWindowsBpm(bmpImage);
    return 0;
}
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "bmp_image_util.h"

void otsu(uchar* image, const uint WIDTH, const uint HEIGHT) {
      printf("Thread [%d]: Starting Otsu segmentation\n",omp_get_thread_num());

    const uint SIZE = WIDTH*HEIGHT;
    int histogram[256];
    float probabilities[256];
    float cumulativeProbabilities[256];
    int threshHold = 0;

    for (int i=0;i<256;i++) {
        histogram[i] = 0;
        probabilities[i] = 0;
        cumulativeProbabilities[i] = 0; 
    }

    //calculate the histogram
    uint sum = 0;
    for (int i=0;i<SIZE;i++) {
        uchar pixelIntensity = image[i];
        sum+=pixelIntensity;
        histogram[pixelIntensity]++;
    }

    const float mg = ((float) sum)/SIZE;

    float cumulativeProbability = 0;
    for (int i=0;i<256;i++) {
        float probability = ((float)histogram[i])/SIZE;
        probabilities[i] = probability;
        cumulativeProbability+=probability;
        cumulativeProbabilities[i]=cumulativeProbability;
    }

    float maxVariance = 0;
    threshHold = 0;
    int maxCount = 0;
    for (int k=0;k<255;k++) {
        const float pk = cumulativeProbabilities[k];
        float mk = 0;
        for (int i=0;i<=k;i++)
            mk+=i*probabilities[i];
        
        mk/=pk;

        float mk2 = 0;
        for (int i=k+1;i<256;i++)
            mk2+=i*probabilities[i];

        mk2/=(1-pk);

        float variance = pk*(pow(mk-mg,2))+(1-pk)*(pow(mk2-mg,2));

        if (variance>maxVariance) {
            maxVariance = variance;
            maxCount=1;
            threshHold = k;
        }
        else if (variance==maxVariance) {
            threshHold+=k;
            maxCount++;
        }
    }

    threshHold = threshHold/maxCount;


    for (int i=0;i<SIZE;i++) {
        int intensity = *(image+i);
        if (intensity>=threshHold)
            *(image+i) = 255;
        else
            *(image+i) = 0;
    }
    printf("Thread [%d]: The Threshhold is %d\n",omp_get_thread_num(), threshHold);
}

void validate(int argc,char** argv) {
    if (argc!=2) {
        printf("Invalid command. To run the app %s <filename>.bmp\n",argv[0]);
        exit(-1);
    }
}

struct BmpImage performOtsuThresholding(struct BmpImage bmpImage) {
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                #pragma omp task
                otsu(bmpImage.r_channel,bmpImage.image_width,bmpImage.image_height);
                #pragma omp task
                otsu(bmpImage.g_channel,bmpImage.image_width,bmpImage.image_height);
                #pragma omp task
                otsu(bmpImage.b_channel,bmpImage.image_width,bmpImage.image_height);
            }
        }
    }
   
    double end = omp_get_wtime();
    printf("Duration %.5f\n",end-start);
    return bmpImage;
}

int main(int argc, char** argv) {
    validate(argc,argv);
    char* filePath = argv[1];
    struct BmpImage bmpImage = loadWindowsBpm(filePath);
    displayBpmImageMetaData(bmpImage);
    bmpImage = performOtsuThresholding(bmpImage);
    writeBpmImage("./data/otsu_output.bmp",bmpImage);
    disposeWindowsBpm(bmpImage);
    return 0;
}
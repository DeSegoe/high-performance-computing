#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "bmp_image_util.h"

#define PI 3.14159265
#define RATIO 180.0/PI 

float** loadMask(char* fileName, int* WIDTH, int* HEIGHT) {
    float LARGEST_POSSIBLE_KERNEL[32][32];

    int height = 0;
    int width = 0;

    FILE *fp; 
    fp = fopen(fileName, "r");

    int response =1;
    
    char buff[255];
    while (1) {
        response = fscanf(fp, "%s", buff);
        if (response==-1)
            break;
       
        width = 0;
        char * pch;
        pch = strtok (buff,",");
        while (pch!=NULL) {
            LARGEST_POSSIBLE_KERNEL[height][width++] = atof(pch);
            pch = strtok(NULL,",");
        }
        height++;
    }

    fclose(fp);

    *WIDTH = width;
    *HEIGHT = height;

    float** mask = (float**) malloc(sizeof(float*)*height);

    for (int i=0;i<height;i++) {
        mask[i] = (float*) malloc(sizeof(float)*width);
        for (int j=0;j<width;j++) {
            float element = LARGEST_POSSIBLE_KERNEL[i][j];
            mask[i][j] = element;
        }
    }

    return mask;
}

void freeMatrix(float** matrix,int WIDTH,int HEIGHT) {
    for (int i=0;i<HEIGHT;i++)
        free(matrix[i]);
    free(matrix);
}

void applyFilterToImage(float** kernel,float* image, float* outputImage, const uint WIDTH, const uint HEIGHT, const int MASK_WIDTH, const int MASK_HEIGHT) {
    int MASK_OFFSET = MASK_HEIGHT/2;
    
    for (int y=0;y<HEIGHT;y++) {
        for (int x=0;x<WIDTH;x++) {
            int offset = y*WIDTH+x;

            float sum = 0;
            for (int y2=-MASK_OFFSET;y2<=MASK_OFFSET;y2++) {
                for (int x2=-MASK_OFFSET;x2<=MASK_OFFSET;x2++) {
                   int newY = y+y2;
                   int newX = x+x2;
                   if (newX>-1 && newX<WIDTH && newY>-1 && newY<HEIGHT) {
                       sum+=kernel[y2+MASK_OFFSET][x2+MASK_OFFSET]*image[newY*WIDTH+newX];
                   }
                }
            }

            outputImage[offset] = sum;
        }
    }
}

void suppressRange(float* image, uchar MINIMUM,uchar MAXIMUM,const uint WIDTH, const uint HEIGHT) {
    const int SIZE = HEIGHT*WIDTH;
    
    #pragma omp parallel for
    for (int i=0;i<SIZE;i++) {
        int pixelValue = image[i];
        if (pixelValue<MINIMUM)
            image[i] = MINIMUM;
        else if (pixelValue>MAXIMUM)
            image[i] = MAXIMUM;
    }
}

uchar* grayScale(uchar* r_channel,uchar* g_channel,uchar* b_channel,const uint WIDTH, const uint HEIGHT) {
    const int SIZE = WIDTH*HEIGHT;
    uchar* grayscale = (uchar*) malloc(SIZE);
    #pragma omp parallel for
    for (int i=0;i<SIZE;i++) {
        int sum = (int) r_channel[i]+(int)g_channel[i]+(int)+b_channel[i];
        sum/=3;
        grayscale[i] = (uchar) sum;
    }

    return grayscale;
}

void emphasizeEdge(uchar* edges,uchar* r_channel,uchar* g_channel,uchar* b_channel,const uint WIDTH, const uint HEIGHT) {
    const int SIZE = WIDTH*HEIGHT;
    #pragma omp parallel for
    for (int i=0;i<SIZE;i++) {
        if (edges[i]==255) {
            r_channel[i] = g_channel[i] = b_channel[i] = 0;
        }
    }
}

void replaceEdge(uchar* edges,uchar* r_channel,uchar* g_channel,uchar* b_channel,const uint WIDTH, const uint HEIGHT) {
    const int SIZE = WIDTH*HEIGHT;
    #pragma omp parallel for
    for (int i=0;i<SIZE;i++) {
         if (edges[i]==255)
            r_channel[i] = g_channel[i] = b_channel[i] = 255;
        else
            r_channel[i] = g_channel[i] = b_channel[i] = 0;
    }
}

void edgeDetection(uchar* image, const uint WIDTH, const uint HEIGHT) {
    printf("Thread [%d]: Starting Canny edge detection\n",omp_get_thread_num());

    const int SIZE = WIDTH*HEIGHT;
    const int BYTE_SIZE = sizeof(float)*SIZE;
    float* inputImageAsFloat;
    float* smoothedImage;
    float* verticalEdges;
    float* horizontalEdges;
    float* finalResultHighThreshold;
    float* magnitude;
    float* directions;
    uchar* finalResult;

    inputImageAsFloat = (float*) malloc(BYTE_SIZE);
    smoothedImage = (float*) malloc(BYTE_SIZE);
    verticalEdges = (float*) malloc(BYTE_SIZE);
    horizontalEdges = (float*) malloc(BYTE_SIZE);
    magnitude = (float*) malloc(BYTE_SIZE);
    directions = (float*) malloc(BYTE_SIZE);
    finalResultHighThreshold = (float*) malloc(BYTE_SIZE);

    int gausian_width;
    int gausian_height;
    float** gaussianFilter = loadMask("./mask_files/gaussian.txt",&gausian_width,&gausian_height);

    int roberts_width;
    int roberts_height;
    float** robertsHorizontalFilter = loadMask("./mask_files/roberts_horizontal.txt",&roberts_width,&roberts_height);
    float** robertsVerticalFilter = loadMask("./mask_files/roberts_vertical.txt",&roberts_width,&roberts_height);

    for (int k=0;k<SIZE;k++) {
        *(inputImageAsFloat+k) = (float) image[k];
        *(smoothedImage+k) = 0;
        *(verticalEdges+k) = 0;
        *(horizontalEdges+k) = 0;
        *(magnitude+k) = 0;
        *(directions+k) = 0;
        *(finalResultHighThreshold+k) = 0;
    }

    //smooth image using gaussian filter
    applyFilterToImage(gaussianFilter, inputImageAsFloat,smoothedImage,WIDTH,HEIGHT,gausian_width,gausian_height);
    suppressRange(smoothedImage,0,255,WIDTH,HEIGHT);

     //calculate vertical edge
    applyFilterToImage(robertsVerticalFilter, smoothedImage,verticalEdges,WIDTH,HEIGHT,roberts_width,roberts_height);
    //calculate horizontal edge
    applyFilterToImage(robertsHorizontalFilter, smoothedImage,horizontalEdges,WIDTH,HEIGHT,roberts_width,roberts_height);

    //calculate magnitude and direction
    double maximumMagnitude = 0;
    #pragma omp parallel for reduction(max:maximumMagnitude)
    for (int i=0;i<SIZE;i++) {
        float gx = horizontalEdges[i];
        float gy = verticalEdges[i];

        float ithMagnitude = pow(pow(gx,2)+pow(gy,2),0.5);
        magnitude[i] = ithMagnitude;
        directions[i] = atan2(gy,gx)*RATIO;

        if (maximumMagnitude<ithMagnitude) {
            maximumMagnitude = ithMagnitude;
        }
    } 

    float highThresholdValue = maximumMagnitude*0.1;
    //non maxima suppression
     #pragma omp parallel for collapse(2)
    for (int y=1;y<HEIGHT-1;y++) {
        for (int x=1;x<WIDTH-1;x++) {
            int yOffset = y*WIDTH;
            int yMinusOne = (y-1)*WIDTH;
            int yPlusOne = (y+1)*WIDTH;
            int offset = yOffset+x;
            float direction = directions[offset];
            float result = magnitude[offset];
            //horizontal edge
            if ((direction>=-22.5 && direction<=22.5) ||
                (direction>=-180 && direction<=-157.5) ||
                (direction>=157.5 && direction<=180)) {
                    
                if (result < magnitude[offset-1] || result < magnitude[offset+1]) {
                    result = 0;
                }
            }

            //vertial edge
            else if ((direction>=67.5 && direction<=112.5) ||
                (direction>=-112.5 && direction<=-67.5)) {
                    if (result < magnitude[yMinusOne+x] || result < magnitude[yPlusOne+x]) {
                    result = 0;
                }
            }

            //45% edge
            else if ((direction>=112.5 && direction<=157.5) ||
                (direction>=-67.5 && direction<=-22.5)) {
                
                if (result < magnitude[yPlusOne+x-1] || result < magnitude[yMinusOne+x+1]) {
                    result = 0;
                }
            }

            //-45% edge
            else if ((direction>=22.5 && direction<=67.5) ||
                (direction>=-157.5 && direction<=-112.5)) {
                if (result < magnitude[yMinusOne+x-1] || result < magnitude[yPlusOne+x+1]) {
                    result = 0;
                }
            }

        // finalResultHighThreshold[offset]=result;

            if (result>=highThresholdValue)
                finalResultHighThreshold[offset] = result;
            else
                finalResultHighThreshold[offset] = 0;
        }
    } 

    //final transformation
    for (int i=0;i<SIZE;i++) {
        float finalValue = finalResultHighThreshold[i];
        if (finalValue<0)
            image[i] = 0;
        else if (finalValue>255)
            image[i] = 255;
        else
            image[i] = (uchar) finalValue;
    }

    freeMatrix(gaussianFilter,gausian_width,gausian_height);
    freeMatrix(robertsVerticalFilter,roberts_width,roberts_height);
    freeMatrix(robertsHorizontalFilter,roberts_width,roberts_height);
    
    free( inputImageAsFloat);
    free( smoothedImage);
    free( verticalEdges);
    free( horizontalEdges);
    free( magnitude);
    free( directions);
    free(finalResultHighThreshold);
}

void validate(int argc,char** argv) {
    if (argc!=2) {
        printf("Invalid command. To run the app %s <filename>.bmp\n",argv[0]);
        exit(-1);
    }
}

void performEdgeDetectionOnGrayScaleVersionOfColorImage(char* filePath) {
    struct BmpImage bmpImage = loadWindowsBpm(filePath);
     ulong start = omp_get_wtick();
    uchar* grayscale = grayScale(bmpImage.r_channel,bmpImage.g_channel,bmpImage.b_channel,bmpImage.image_width,bmpImage.image_height);
    edgeDetection(grayscale,bmpImage.image_width,bmpImage.image_height);
    emphasizeEdge(grayscale,bmpImage.r_channel,bmpImage.g_channel,bmpImage.b_channel,bmpImage.image_width,bmpImage.image_height);
    writeBpmImage("./data/sharpened_output.bmp",bmpImage);
    replaceEdge(grayscale,bmpImage.r_channel,bmpImage.g_channel,bmpImage.b_channel,bmpImage.image_width,bmpImage.image_height);
    writeBpmImage("./data/edge_output.bmp",bmpImage);
   
    ulong end = omp_get_wtick();
    printf("Duration %u\n",end-start);
    free(grayscale);
    disposeWindowsBpm(bmpImage);
}

uchar* createImageCopy(uchar* original,const uint size) {
    uchar* copy = (uchar*) malloc(size);
    memcpy(copy,original,size);
    return copy;
}

void edgeDetectionOnColorImage(char* filePath) {
    struct BmpImage bmpImage = loadWindowsBpm(filePath);
    double start = omp_get_wtime();
    int size = bmpImage.image_width*bmpImage.image_height;
    uchar* r_channel = createImageCopy(bmpImage.r_channel,size);
    uchar* g_channel = createImageCopy(bmpImage.g_channel,size);
    uchar* b_channel = createImageCopy(bmpImage.b_channel,size);

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                #pragma omp task
                edgeDetection(bmpImage.r_channel,bmpImage.image_width,bmpImage.image_height);
                #pragma omp task
                edgeDetection(bmpImage.g_channel,bmpImage.image_width,bmpImage.image_height);
                #pragma omp task
                edgeDetection(bmpImage.b_channel,bmpImage.image_width,bmpImage.image_height);
            }
        }
    }

    writeBpmImage("./data/color_edges.bmp",bmpImage);

    #pragma omp parallel for
    for (int i=0;i<size;i++) {
        bmpImage.r_channel[i] = bmpImage.r_channel[i]==255? 0:r_channel[i];
        bmpImage.g_channel[i] = bmpImage.g_channel[i]==255? 0:g_channel[i];
        bmpImage.b_channel[i] = bmpImage.b_channel[i]==255? 0:b_channel[i];
    }

    writeBpmImage("./data/sharpened_using_color_edges.bmp",bmpImage);

    double end = omp_get_wtime();
    printf("Duration %.8f\n",end-start);
    free(r_channel);
    free(g_channel);
    free(b_channel);
    disposeWindowsBpm(bmpImage);
}

int main(int argc, char** argv) {
    validate(argc,argv);
    char* filePath = argv[1];
   
    performEdgeDetectionOnGrayScaleVersionOfColorImage(filePath);
    edgeDetectionOnColorImage(filePath);

    return 0;
}
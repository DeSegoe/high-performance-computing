#ifndef BPM_IMAGE_UTIL
#define BPM_IMAGE_UTIL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 1 << 7
#define BMP_HEADER 54

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

struct BmpImage {
    char* filePath;
    char image_type[2];

    int file_size;
    int image_data_size;
    int image_address;
    int image_width;
    int image_height;
    int colors;
    int bits_per_pixel;
    int palette;
    uchar* bitmap_data;
    uchar* r_channel;
    uchar* g_channel;
    uchar* b_channel;
    char* header;
    
};


struct BmpImage loadWindowsBpm(char* filePath) {
    struct BmpImage bmpImage;
    bmpImage.filePath = filePath;
    FILE* f = fopen(filePath, "rb");

    if(f == NULL) {
      printf("Error, could not open file %s\n",filePath);   
      exit(1);
    }

    bmpImage.header = (uchar*) malloc(BMP_HEADER);
    fread(bmpImage.header, sizeof(uchar), BMP_HEADER, f); // read the 54-byte header
    memcpy(bmpImage.image_type,&bmpImage.header[0],2);
    memcpy(&bmpImage.file_size,&bmpImage.header[2],4);
    memcpy(&bmpImage.image_address,&bmpImage.header[10],4);
    memcpy(&bmpImage.image_width,&bmpImage.header[18],4);
    memcpy(&bmpImage.image_height,&bmpImage.header[22],4);
    memcpy(&bmpImage.colors,&bmpImage.header[26],2);
    memcpy(&bmpImage.bits_per_pixel,&bmpImage.header[28],2);
    memcpy(&bmpImage.palette,&bmpImage.header[46],4);

    const int _3_size = 3 * bmpImage.image_width * bmpImage.image_height;
    
    if (_3_size>bmpImage.file_size) {
        printf("Invalid bpm file %s has been specified\n",filePath);
        fclose(f);
        exit(-1);
    }

    bmpImage.image_data_size = _3_size;
    
    bmpImage.bitmap_data = (uchar*) malloc(bmpImage.file_size-BMP_HEADER); // allocate 3 bytes per pixel
    fread(bmpImage.bitmap_data, sizeof(unsigned char), bmpImage.file_size, f); // read the rest of the data at once
    fclose(f);

    // for(int i = 0; i < _3_size; i += 3)
    // {
    //     uchar tmp = bmpImage.bitmap_data[i];
    //     bmpImage.bitmap_data[i] = bmpImage.bitmap_data[i+2];
    //     bmpImage.bitmap_data[i+2] = tmp;
    // }

    bmpImage.r_channel = (uchar*) malloc(bmpImage.image_width * bmpImage.image_height);
    bmpImage.g_channel = (uchar*) malloc(bmpImage.image_width * bmpImage.image_height);
    bmpImage.b_channel = (uchar*) malloc(bmpImage.image_width * bmpImage.image_height);

    //load difference channels
    for (int y=0;y<bmpImage.image_height;y++) {
        for (int x=0;x<bmpImage.image_width;x++) {
            int index = y*bmpImage.image_width+ x;

            bmpImage.g_channel[index] = bmpImage.bitmap_data[3*(index)] ;
            bmpImage.r_channel[index] = bmpImage.bitmap_data[3*(index)+1] ;
            bmpImage.b_channel[index] = bmpImage.bitmap_data[3*(index)+2] ;
        }
    }

    return bmpImage;
}

void writeBpmImage(char* filePath, struct BmpImage bmpImage) {
    FILE * f = fopen(filePath,"wb");

    int number = 25;

    if (f == NULL) {
        printf("Could not write to file %s\n",filePath);
        exit(-1);
    }

    //load difference channels
    for (int y=0;y<bmpImage.image_height;y++) {
        for (int x=0;x<bmpImage.image_width;x++) {
            int index = y*bmpImage.image_width+ x;

           bmpImage.bitmap_data[3*(index)]   = bmpImage.g_channel[index] ;//green
           bmpImage.bitmap_data[3*(index)+1] = bmpImage.r_channel[index] ;//red
           bmpImage.bitmap_data[3*(index)+2] = bmpImage.b_channel[index];//blue
            //bmpImage.bitmap_data[3*(index)]   = 0;//green
            //bmpImage.bitmap_data[3*(index)+1] = 0;//red...
            //bmpImage.bitmap_data[3*(index)+2] = 0;//blue...
        }
    }

    fwrite(bmpImage.header,BMP_HEADER,1,f);
    fwrite(bmpImage.bitmap_data,bmpImage.file_size-BMP_HEADER,1,f);

    fclose(f);

    printf("Finished saving %s\n",filePath);
}


void displayBpmImageMetaData(struct BmpImage bmpImage) {
    printf("Width:%d, Height:%d, File size: %d, Image size: %d,Channels:%d, Bits per pixel: %d, Colour palette:%d\n",bmpImage.image_width,bmpImage.image_height,bmpImage.file_size,bmpImage.image_data_size, bmpImage.colors,bmpImage.bitmap_data,bmpImage.palette);
}

void disposeWindowsBpm(struct BmpImage bmpImage) {
    free(bmpImage.bitmap_data);
    free(bmpImage.r_channel);
    free(bmpImage.g_channel);
    free(bmpImage.b_channel);
    free(bmpImage.header);
}

#endif

//https://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file
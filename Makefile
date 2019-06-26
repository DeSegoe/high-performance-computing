CC=gcc
OMP_FLAG=-fopenmp
MATH_FLAG=-lm

all: otsu_segmentation

otsu_segmentation: otsu_segmentation.c bmp_image_util.h
	$(CC) $(OMP_FLAG) $(MATH_FLAG) otsu_segmentation.c -o otsu_segmentation.exe
	echo Success!!!

clean:
	rm -Force otsu_segmentation.exe
	
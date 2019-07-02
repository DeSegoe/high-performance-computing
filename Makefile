CC=gcc
NVCC=nvcc
OMP_FLAG=-fopenmp
CUDA_OMP_FLAG=-Xcompiler="-openmp"
MATH_FLAG=-lm

all: otsu_segmentation canny_detection cluster_segmentation

otsu_segmentation: otsu_segmentation.c bmp_image_util.h
	$(CC) $(OMP_FLAG) $(MATH_FLAG) otsu_segmentation.c -o otsu_segmentation.exe
	echo Successfully built the otsu segmentation application!!!

canny_detection: canny_edge_detection.c bmp_image_util.h
	$(CC) $(OMP_FLAG) $(MATH_FLAG) canny_edge_detection.c -o canny_edge_detection.exe
	echo Successfully built the canny edge detection application!!!

cluster_segmentation: cluster_segmentation.c bmp_image_util.h
	$(CC) $(OMP_FLAG) $(MATH_FLAG) cluster_segmentation.c -o cluster_segmentation.exe
	echo Successfully built the cluster segmentation application!!!

aggregates:aggregate_operations.c
	$(CC) $(OMP_FLAG) $(MATH_FLAG) aggregate_operations.c -o aggregate_operations.exe

cuda_dev:test_cuda_context.cu cuda.h
	$(NVCC) $(CUDA_OMP_FLAG) .\test_cuda_context.cu -o test_cuda_context.exe

clean:
	rm -Force otsu_segmentation.exe canny_edge_detection.exe cluster_segmentation.exe aggregate_operations.exe cuda_aggregate_operations.exe
	
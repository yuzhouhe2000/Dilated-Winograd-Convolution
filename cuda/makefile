
all: conv2d_gpu

conv2d_gpu: conv2d_gpu.o utils_cu.o im2col_cu.o
	nvcc -o  conv2d_gpu conv2d_gpu.o utils_cu.o im2col_cu.o

conv2d_gpu.o: conv2d_gpu.cu utils_cu.h im2col_cu.h
	nvcc -c conv2d_gpu.cu 

utils_cu.o: utils_cu.cu utils_cu.h 
	nvcc -c utils_cu.cu

im2col_cu.o: im2col_cu.cu im2col_cu.h utils_cu.h
	nvcc -c im2col_cu.cu

clean: 
	-rm -rf *.o conv2d_gpu *.out *.err


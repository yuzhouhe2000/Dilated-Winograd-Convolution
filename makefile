
all: conv2d_cpu 

conv2d_cpu: conv2d_cpu.o winograd_transform.o im2col.o utils.o
	gcc -lrt -fopenmp -o conv2d_cpu conv2d_cpu.o winograd_transform.o im2col.o utils.o

conv2d_cpu.o: conv2d_cpu.c utils.h im2col.h winograd_transform.h
	gcc -lrt -fopenmp -c conv2d_cpu.c 


winograd_transform.o: winograd_transform.c winograd_transform.h utils.h
	gcc -lrt -fopenmp -c winograd_transform.c


im2col.o: im2col.c im2col.h utils.h
	gcc -lrt -fopenmp -c im2col.c


utils.o: utils.c utils.h
	gcc -lrt -fopenmp -c utils.c

clean: 
	-rm -rf *.o conv2d_cpu


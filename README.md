# Dilated-Convolution

Implementation of GPU and CPU based convolution with dilation, including direct convolution, im2col GEMM, winograd transformation.

To run CPU version:

    cd to the "cpu" folder
    make all
    ./conv2d_cpu

To run GPU version:

    cd to the "cuda" folder
    make all
    sbatch job.sl
    vim conv2d_gpu.out

NOTE:

The CPU winograd only support dilation = 2. It takes a 8x8 input tile and break it into four 7x7 tile for a winograd with dilated kernel. The output is a 4x4 tile merged from four 3x3 tiles. It makes parallelization difficult, so in the GPU version, I sample the winograd input in a dilated fashion and use the original 3x3 kernel for winograd instead of 5x5. As a result, in my GPU dilated winograd implementation, the input tile size and output size will be the same as a normal F(2x2,3x3) winograd.


TODO:

- [ ] parameter vs. performance graph
- [ ] project report
- [ ] CuDNN baseline
- [ ] CUDA accelerated im2col


DONE:

- [x] Overall structure
- [x] CPU direct convolution
- [x] CPU im2col
- [x] CPU dilated winograd
- [x] CUDA accelerated direct convolution
- [x] CUDA accelerated dilated winograd
- [x] improve CUDA accelerated winograd 
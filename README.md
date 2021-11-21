# Dilated-Convolution

Implementation of GPU and CPU based convolution with dilation, including direct convolution, im2col GEMM, winograd transformation.

To run CPU version:

    make all
    ./conv2d_cpu

To run GPU version:

    make all
    sbatch job.sl

TODO:

- [ ] CUDA accelerated dilated winograd (priority)
- [ ] CUDA accelerated im2col
- [ ] CuDNN baseline
- [ ] More Winograd kernel shapes, dilation rates and stride

DONE:

- [x] Overall structure
- [x] CPU direct convolution
- [x] CPU im2col
- [x] CPU dilated winograd
- [x] CUDA accelerated direct convolution
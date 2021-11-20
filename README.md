# Dilated-Convolution

Implementation of GPU and CPU based convolution with dilation, including direct convolution, im2col GEMM, winograd transformation.

To run CPU version:

    make all
    ./conv2d_cpu

TODO:

- [ ] CUDA accelerated direct convolution
- [ ] CUDA accelerated im2col
- [ ] CUDA accelerated dilated winograd
- [ ] CuDNN baseline
- [ ] More Winograd kernel shapes, dilation rates and stride


DONE:

- [x] Overall structure
- [x] CPU direct convolution
- [x] CPU im2col
- [x] CPU dilated winograd
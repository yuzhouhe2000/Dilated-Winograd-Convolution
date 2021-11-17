# Dilated-Convolution

Implementation of GPU and CPU based convolution with dilation, including direct convolution, im2col GEMM, winograd transformation.

TODO:

- [ ] openMP accelerated direct convolution
- [ ] CUDA accelerated direct convolution
- [ ] openMP accelerated im2col
- [ ] CUDA accelerated im2col
- [ ] openMP accelerated dilated winograd
- [ ] CUDA accelerated dilated winograd
- [ ] CuDNN baseline
- [ ] More Winograd kernel shapes and dilation rates


DONE:

- [x] Overall structure
- [x] Naive direct convolution
- [x] Naive im2col
- [x] Naive dilated winograd
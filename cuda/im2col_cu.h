float* im2col_dilated_cpu(float* data_im,int channels, int height, int width,int ksize, int stride, int pad, int dilate_rate);
float im2col_get_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad);
void col2im_add_pixel_dilated(float *im, int height, int width, int channels,int row, int col, int channel, int pad, float val);
float* col2im_dilated_cpu(float* data_col,int channels,  int height,  int width,int ksize,  int stride, int pad, int dilate_rate);
float* im2col_mm(float* A, float* B,int X,int C ,int Hk,int Wk, int Z,int dilH,int dilW);
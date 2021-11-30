//https://github.com/BVLC/caffe/
//im2col and col2im modified from caffe framework 

#include "im2col.h"
#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <stdlib.h>


float im2col_get_pixel(float *im, int height, int width, int channels,int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

float* im2col_dilated_cpu(float* data_im,int channels,  int height,  int width,int ksize,  int stride, int pad, int dilate_rate) {
    int c,h,w;
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2*pad - dilate_ksize) / stride + 1;
    int width_col = (width + 2*pad - dilate_ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    float* data_col = (float*)malloc(sizeof(float) * width_col*height_col*channels_col);
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize + 1;
        int h_offset = (c / ksize) % ksize + 1;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilate_rate + h * stride - 1;
                int im_col = w_offset * dilate_rate + w * stride - 1;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,im_row, im_col, c_im, pad);
            }
        }
    }
    
    return data_col;
}


void col2im_add_pixel_dilated(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if ((row-1) < 0 || (col-1) < 0 ||
        (row-1) >= height || (col-1) >= width){
            return;
        }
    im[col-1 + width*(row-1 + height*channel)] += val;
}


float* col2im_dilated_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, int dilate_rate){
    int c,h,w;
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2*pad - dilate_ksize) / stride + 1;
    int width_col = (width + 2*pad - dilate_ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    float* data_im = (float*)malloc(sizeof(float) * height * width * channels);
    for (c = 1; c <= channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize+1;
        if (w_offset == 0){
        	w_offset = ksize;
        	h_offset--;
        }
        if (h_offset == 0) h_offset = ksize;
        int c_im = (c-1) / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilate_rate + h * stride;
                int im_col = w_offset * dilate_rate + w * stride;
                int col_index = ((c-1) * height_col + h) * width_col + w;
                double val = data_col[col_index];
                
                printf("%d ", col_index);
                //printf("im_row = %d, im_col = %d, val = %d\t location in window:(%d, %d)\n",im_row, im_col, (int)val, h_offset, w_offset);
                col2im_add_pixel_dilated(data_im, height, width, channels,im_row, im_col, c_im, pad, val);
            }
            printf("\n");
        }
        printf("\n");
    }
    return data_im;

}


float* im2col_mm(float* A, float* B,int X,int Cin ,int Hk,int Wk, int Z, int dilH, int dilW){
    float* C = (float*)malloc(sizeof(float) * X * Z);
    int Y = Cin*Hk*Wk;
    int x;
    // #pragma omp parallel for
    for (x = 0; x < X; x++) {
        for (int z = 0; z < Z; z++) {
            C[x * Z + z] = 0.0f;
            for (int cin = 0; cin < Cin; cin++) {
                for (int hk = 0; hk < Hk; hk=hk+dilH) {
                    for (int wk = 0; wk < Wk; wk=wk+dilW) {
                        int y = cin*Hk*Wk + hk*Wk+wk;
                        C[x * Z + z] += A[x * Y + y] * B[y * Z + z];
                    }
                }
            }
        }
    }
    return C;
}
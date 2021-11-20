#include <stdlib.h>
#include "utils.h"
#include<stdio.h>
#include<string.h>
#include <math.h>

// Safe free
void free_(float* ptr){
	if (ptr) {
		free(ptr);
		ptr = NULL;
	}
}

void fill(float* input,int SIZE){
	for(int i = 0;i<SIZE;i++){
		input[i] = 0.0f;
	}
}

int check_tensor(struct tensor_ A, struct tensor_ B) {
	if (A.SIZE != B.SIZE) {
		printf("Output SIZE Incorrect!\n");
		return 0;
	}
	for (int i = 0; i < A.SIZE; i++) {
		// accept if error < 0.01 * true value
		if (fabs(A.data[i] - B.data[i]) > fabs(A.data[i]*0.01)){
			printf("Output ELEMENT Incorrect!\n");
			return 0;
		}
	}
	printf("Output Correct!\n");
	return 1;
}

float* transpose(float *input, const int N,const int C,const int H, const int W) {
	float* inputT = (float*)malloc(sizeof(float) * N*C*H*W);
	int n;
	#pragma omp parallel for
	for(n = 0; n<N; n++) {
		for(int c = 0; c <C; c++) {
			for(int h = 0; h<H; h++) {
				for (int w = 0; w < W; w++) {
					inputT[n * C * H * W + c * H * W + w*H+h] = input[n * C * H * W + c * H * W + W * h + w];
				}
			}
		}
	}
	return inputT;
}

float* NCHW_2_NHWC(float* input, const int N, const int C, const int H, const int W) {
	float* inputT = (float*)malloc(sizeof(float) * N * C * H * W);
	int n;
	#pragma omp parallel for
	for (n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					inputT[find_NCHW_idx(n,h,w,c,N,H,W,C)] = input[find_NCHW_idx(n, c, h, w, N, C, H, W)];
				}
			}
		}
	}
	return inputT;
}

float* NHWC_2_NCHW(float* input, const int N, const int C, const int H, const int W) {
	float* inputT = (float*)malloc(sizeof(float) * N * C * H * W);
	int n;
	#pragma omp parallel for
	for (n = 0; n < N; n++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				for (int c = 0; c < C; c++) {
					inputT[find_NCHW_idx(n, c, h, w, N, C, H, W)] = input[find_NCHW_idx(n, h, w, c, N, H, W, C)];
				}
			}
		}
	}
	return inputT;
}


float* slice(float* input,int start,int end){
	float* input_slice = (float*)malloc(sizeof(float) * (end-start));
	memcpy(input_slice, input + start, (end - start) * sizeof(float));
	return input_slice;
}

int find_kernel_idx(int cout,int cin,int hk, int wk,struct kernel_ kernel){
	return cout * kernel.Cin * kernel.H * kernel.W + cin * kernel.H * kernel.W + hk * kernel.W + wk;
}

int find_tensor_idx(int n,int cin, int hin,int win,struct tensor_ input){
	return n * input.H * input.W * input.C + cin*input.W*input.H + hin * input.W  + win;
}

int find_NCHW_idx(int n,int cin,int hin,int win,int N,int C,int H,int W){
	return n * H * W * C + cin*H*W +  hin * W + win;
}

int find_CCHW_idx(int cout,int cin,int hk,int wk,int Cout,int Cin,int H,int W){
	return cout * Cin * H * W + cin * H * W + hk * W + wk;
}

void print_kernel(struct kernel_ kernel) {
	int kernel_idx;
	for (int cout = 0; cout < kernel.Cout; cout++) {
		for (int cin = 0; cin < kernel.Cin; cin++) {
			for (int hk = 0; hk < kernel.H; hk++) {
				for (int wk = 0; wk < kernel.W; wk++) {
					kernel_idx = find_kernel_idx(cout,cin,hk,wk,kernel);
					printf("[%.f] ", kernel.data[kernel_idx]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}

}

void print_CHW(float* input,int C,int H,int W) {
	int kernel_idx;
	for (int cin = 0; cin < C; cin++) {
		for (int hk = 0; hk < H; hk++) {
			for (int wk = 0; wk < W; wk++) {
				kernel_idx = cin*H*W + hk*W + wk;

				printf("[%.2f] ", input[kernel_idx]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void print_W(float* input,int W) {
	for (int wk = 0; wk < W; wk++) {
		printf("[%.f] ", input[wk]);
	}
	printf("\n");
	printf("\n");
}

void print_tensor(struct tensor_ output) {
	for (int n = 0; n < output.N; n++) {
		for (int cout = 0; cout < output.C; cout++) {
			for (int hout = 0; hout < output.H; hout++) {
				for (int wout = 0; wout < output.W; wout++) {
					printf("[%.f] ", output.data[find_tensor_idx(n,cout,hout,wout,output)]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

// naive kernel dilation transformation on CPU
struct kernel_ kernel_simple_dilation(struct kernel_ kernel){
	if ((kernel.dilH == 1)&&(kernel.dilW == 1)){
		return kernel;
	}
	int posH, posW, kernel_idx, kernel_idx_new;
	// // dilation method 1: 0101010    dil(3,2) = 7
	// int newH = (kernel.dilH - 1) * (kernel.H + 1) + kernel.H;
	// int newW = (kernel.dilW - 1) * (kernel.W + 1) + kernel.W;
	// dilation method 2: 10101		dil(3,2) = 5
	int newH = (kernel.dilH-1) * (kernel.H-1)+kernel.H;
	int newW = (kernel.dilW-1) * (kernel.W-1)+kernel.W;
	int newSize = kernel.Cout * newH * newW * kernel.Cin;
	float* B_dil = (float*)malloc(sizeof(float) * newSize);
	for (int i = 0; i < newSize; i++) {
		B_dil[i] = 0.0f;
	}
	for (int cout = 0; cout < kernel.Cout;cout++){
		for (int cin = 0; cin < kernel.Cin;cin++){
			for (int hk = 0; hk < kernel.H;hk++){
				for (int wk = 0; wk < kernel.W;wk++){
					// // dilation 1
					// posH = ((hk+1)*kernel.dilH-1);
					// posW = ((wk+1)*kernel.dilW-1);
					// dilation 2
					posH = hk*kernel.dilH;
					posW = wk*kernel.dilW;
					kernel_idx = find_kernel_idx(cout,cin,hk,wk,kernel);
					kernel_idx_new = find_CCHW_idx(cout,cin,posH,posW,kernel.Cout,kernel.Cin,newH,newW);
					B_dil[kernel_idx_new] = kernel.data[kernel_idx];
				}
			}
		}
	}
	struct kernel_ kernel_new = { .data = B_dil, .Cout = kernel.Cout, .Cin = kernel.Cin, .H = newH, .W = newW, .dilH = 1, .dilW = 1, .padH = kernel.padH, .padW = kernel.padW, .strideH = kernel.strideH, .strideW = kernel.strideW,.SIZE = newSize};
	//print_kernel(kernel_new);
	return kernel_new;
} 



// pad tensor
struct tensor_ tensor_pad(struct tensor_ input,int padH,int padW){
	if ((padH == 0)&&(padW == 0)){
		return input;
	}
	int kernel_idx;
	int newH = input.H+2*padH;
	int newW = input.W+2*padW;
	int newSize = input.C * newH * newW * input.N;
	float* output = (float*)malloc(sizeof(float) * newSize);
	for (int n = 0; n < input.N; n++) {
		for (int hout = 0; hout < newH; hout++) {
			for (int wout = 0; wout < newW; wout++) {
				for (int cout = 0; cout < input.C; cout++) {
					if (hout < padH || hout >= (newH-padH)|| wout < padW || wout >= (newW - padW)){
						output[find_NCHW_idx(n,cout,hout,wout,input.N,input.C,newH,newW)] = 0;
					}
					else{
						kernel_idx = find_tensor_idx(n,cout,hout-padH,wout-padW,input);
						output[find_NCHW_idx(n, cout, hout, wout, input.N, input.C, newH, newW)] = input.data[kernel_idx];
					}
				}
			}
		}
	}
	struct tensor_ padded = { .data = output, .H = newH, .W = newW, .N = input.N, .C = input.C,.SIZE = newH*newW*input.N*input.C};
	print_tensor(padded);
	return padded;
} 


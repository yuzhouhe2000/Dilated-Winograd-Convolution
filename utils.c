#include <stdlib.h>
#include <utils.h>
#include<stdio.h>

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

				printf("[%.f] ", input[kernel_idx]);
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
	// int newH = (kernel.H-1)*(2*kernel.dilH-1)+1;
	// int newW = (kernel.W-1)*(2*kernel.dilW-1)+1;
	int newH = (kernel.dilH - 1) * (kernel.H + 1) + kernel.H;
	int newW = (kernel.dilW - 1) * (kernel.W + 1) + kernel.W;
	int newSize = kernel.Cout * newH * newW * kernel.Cin;
	float* B_dil = (float*)malloc(sizeof(float) * newSize);
	for (int i = 0; i < newSize; i++) {
		B_dil[i] = 0.0;
	}
	for (int cout = 0; cout < kernel.Cout;cout++){
		for (int cin = 0; cin < kernel.Cin;cin++){
			for (int hk = 0; hk < kernel.H;hk++){
				for (int wk = 0; wk < kernel.W;wk++){
					posH = ((hk+1)*kernel.dilH-1);
					posW = ((wk+1)*kernel.dilW-1);
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

	
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "utils_cu.h"

// Golden Output. Naive implementation of direct convolution on CPU. Tensor in NCHW layout.
struct tensor_ conv2d_direct_convolution_cpu(struct tensor_ input, struct kernel_ kernel_raw){
	struct kernel_ kernel = kernel_simple_dilation(kernel_raw);
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;
	float* C = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	int n;
	for (n = 0; n < input.N;n++){
		for (int cout = 0; cout < kernel.Cout; cout++){
			for (int hout = 0; hout < Hout; hout++){
				for (int wout = 0; wout < Wout; wout++){
					float accum = 0.0f;
					for (int cin = 0; cin < input.C;cin++){
						for (int hk = 0; hk < kernel.H;hk=hk+kernel.dilH){
							for (int wk = 0; wk < kernel.W;wk=wk+kernel.dilW){	
								int hin = (hout * kernel.strideH + hk)-kernel.padH;
								int win = (wout * kernel.strideW + wk)-kernel.padW;
								if (hin < 0 || hin >= input.H || win < 0 || win >= input.W) {
									accum += 0;
								}
								else {
									int input_idx = find_tensor_idx(n, cin, hin, win, input);
									int kernel_idx = find_kernel_idx(cout, cin, hk, wk, kernel);
									accum += (input.data[input_idx] * kernel.data[kernel_idx]);	
								}
							}
						}
					}
					C[find_NCHW_idx(n,cout,hout,wout,input.N,kernel.Cout,Hout,Wout)] = accum;
				}
			}
		}
	}
	free(kernel.data);
	struct tensor_ output = { .data = C,.N = input.N, .H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	return output;
}

// Naive implementation of direct convolution on CPU. Tensor in NCHW layout.
__global__ void conv2d_direct_convolution_gpu(struct tensor_ input, struct kernel_ kernel, struct tensor_ output){
	// Distribute computation of each output pixel to a separate cuda kernel
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int n = idx/(kernel.Cout * output.H * output.W);
	int cout = ((idx% (kernel.Cout * output.H * output.W))/(output.H * output.W)) ;
	int hout = ((idx% (output.H*output.W))/(output.W)) ;
	int wout = idx%(output.W);

	// TODO: use shared memory
	float accum = 0.0f;
	for (int cin = 0; cin < input.C;cin++){
		for (int hk = 0; hk < kernel.H;hk++){
			for (int wk = 0; wk < kernel.W;wk++){	
				int hin = (hout * kernel.strideH + hk*kernel.dilH)-kernel.padH;
				int win = (wout * kernel.strideW + wk*kernel.dilW)-kernel.padW;
				int input_idx = n * input.H * input.W * input.C + cin*input.W*input.H + hin * input.W  + win;
				int kernel_idx = cout * kernel.Cin * kernel.H * kernel.W + cin * kernel.H * kernel.W + hk * kernel.W + wk;
				accum += (input.data[input_idx] * kernel.data[kernel_idx]);	
				
			}
		}
	}
	output.data[n * kernel.Cout * output.H * output.W + cout*output.H*output.W +  hout * output.W + wout] = accum;
}

__global__ void wino23s1d2_GgGT_kernel(struct kernel_ kernel,float* Gg,float* GgGT){
	// (4x3),(3x3) -> (4,3)
	// Sparse
	for (int cout = 0; cout < kernel.Cout; ++cout){
		for (int cin = 0; cin < kernel.Cin; ++cin){
			for (int j = 0; j < 3; j++) {
				Gg[cout*kernel.Cin*4*3+cin*4*3+0*3+j] = kernel.data[cout*kernel.Cin*3*3+cin*3*3+0*3+j];
				Gg[cout*kernel.Cin*4*3+cin*4*3+1*3+j] = 0.5f*kernel.data[cout*kernel.Cin*3*3+cin*3*3+0*3+j]+
														0.5f*kernel.data[cout*kernel.Cin*3*3+cin*3*3+1*3+j]+
														0.5f*kernel.data[cout*kernel.Cin*3*3+cin*3*3+2*3+j];
				Gg[cout*kernel.Cin*4*3+cin*4*3+2*3+j] = 0.5f*kernel.data[cout*kernel.Cin*3*3+cin*3*3+0*3+j]-
														0.5f*kernel.data[cout*kernel.Cin*3*3+cin*3*3+1*3+j]+
														0.5f*kernel.data[cout*kernel.Cin*3*3+cin*3*3+2*3+j];
				Gg[cout*kernel.Cin*4*3+cin*4*3+3*3+j] = kernel.data[cout*kernel.Cin*3*3+cin*3*3+2*3+j];
			}
		}
	}

	// (4x3),(3x4) -> (4,4)
	for (int cout = 0; cout < kernel.Cout; ++cout){
		for (int cin = 0; cin < kernel.Cin; ++cin){
			for (int i = 0; i < 4; ++i) {
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+0] = Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+0];
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+1] = 0.5f*Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+0]+
														0.5f*Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+1]+
														0.5f*Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+2];
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+2] = 0.5f*Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+0]-
														0.5f*Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+1]+
														0.5f*Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+2];
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+3] = Gg[cout*kernel.Cin*4*3+cin*4*3+i*3+2];
			}
		}
	}
}

__global__ void wino23s1d2_BTxB_EWMM_ATMA_kernel(struct tensor_ input,struct tensor_ output,float* U,int dil){
	// Assume only 1 CUDA kernel,dilation = 2
	// TODO: extend to dilation = 4
	
	// Distribute computation of each output pixel to a separate cuda kernel
	// Original structure:

	// for (int n = 0; n < input.N;n++){
	// 	// for each cout
	// 	for (int cout = 0; cout <output.C; cout++){
	// 		// for each tile
	// 		for (int h = 0; h < input.H-dil*2;h=h+dil*2){
	// 			for (int w = 0; w < input.W-dil*2;w=w+dil*2){
	// 				for (int hh = 0; hh < dil; hh++){
	// 					for (int ww = 0; ww < dil; ww++){
	// 						// define tiles

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int n = idx/(output.C * ((input.H-dil*2)/(dil*2)) *((input.W-dil*2)/(dil*2)) * dil * dil);
	int cout = (idx%(output.C*((input.H-dil*2)/(dil*2))*((input.W-dil*2)/(dil*2))*dil*dil))/(((input.H-dil*2)/(dil*2))*((input.W-dil*2)/(dil*2))*dil*dil);
	int h = (idx%(((input.H-dil*2)/(dil*2))*((input.W-dil*2)/(dil*2))*dil*dil))/(((input.W-dil*2)/(dil*2))*dil*dil)*(dil*2);
	int w = ((idx%(((input.W-dil*2)/(dil*2))*dil*dil))/(dil*dil))*(dil*2);
	int hh = (idx%(dil*dil))/(dil);
	int ww = idx%dil;

	// // parallel
	float accum;
	float* dilated_input = (float*)malloc(sizeof(float)*input.C*4*4);
	float* BTx = (float*)malloc(sizeof(float)*input.C*4*4);
	float* V = (float*)malloc(sizeof(float)*input.C*4*4);
	float* M = (float*)malloc(sizeof(float)*4*4);
	float* ATM = (float*)malloc(sizeof(float)*4*4);
	float* ATMA = (float*)malloc(sizeof(float)*4*4);
	// channels of the tile
	for (int cin = 0; cin < input.C; cin++){ 
		// for each 4x4 dilated input matrix
		// start H and W position of the input tile
		// inside the tile
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				int posH = h+hh+i*dil;
				int posW = w+ww+j*dil;
				dilated_input[cin*16+i*4+j] = input.data[n*input.H*input.W*input.C+cin*input.H*input.W+posH*input.W+posW];
			}
		}

		// use 4x4 input matrix with cin channels to calculate BTxB
		// Sparse
		for (int j = 0; j < 4; j++) {
			BTx[cin * 16 + 0 * 4 + j] = dilated_input[cin*16+0*4+j] - dilated_input[cin*16+2*4+j];
			BTx[cin * 16 + 1 * 4 + j] = dilated_input[cin*16+1*4+j] + dilated_input[cin*16+2*4+j];
			BTx[cin * 16 + 2 * 4 + j] = - dilated_input[cin*16+1*4+j] + dilated_input[cin*16+2*4+j];
			BTx[cin * 16 + 3 * 4 + j] = dilated_input[cin*16+1*4+j] - dilated_input[cin*16+3*4+j];
		}

		// (4x4),(4x4) -> (4,4)
		for (int i = 0; i < 4; ++i){
			V[cin * 16 + i * 4 + 0] = BTx[cin*16+i*4+0] - BTx[cin*16+i*4+2];
			V[cin * 16 + i * 4 + 1] = BTx[cin*16+i*4+1] + BTx[cin*16+i*4+2];
			V[cin * 16 + i * 4 + 2] = - BTx[cin*16+i*4+1] + BTx[cin*16+i*4+2];
			V[cin * 16 + i * 4 + 3] = BTx[cin*16+i*4+1] - BTx[cin*16+i*4+3];
		}
	}

	// compute M matrix for one output channel
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			accum = 0.0f;
			// TODO: maybe HWC instead of CHW? or change algorithm?
			for (int cin = 0; cin < input.C; ++cin){
				accum += U[cout*input.C*16+cin*16+i*4+j] * V[cin*16+i*4+j];
			}
			M[i*4+j] = accum;
		}
	}

	// ATMA
	// (2x4),(4x4) -> (2,4)
	for (int j = 0; j < 4; j++) {
		ATM[0*4+j] = M[0*4+j]+M[1*4+j]+M[2*4+j];
		ATM[1*4+j] = M[1*4+j]-M[2*4+j]-M[3*4+j];
	}
	
	// (2x4),(4x2) -> (2,2)
	for (int i = 0; i < 2; i++) {
		ATMA[i*2 + 0] = ATM[i*4+0]+ATM[i*4+1]+ATM[i*4+2];
		ATMA[i*2 + 1] = ATM[i*4+1]-ATM[i*4+2]-ATM[i*4+3];
	}

	// // map ATMA to the output matrix:
	int H_start = h+hh;
	int W_start = w+ww;
	output.data[n*output.H*output.W*output.C+cout*output.H*output.W + H_start*output.W + W_start] = ATMA[0];
	output.data[n*output.H*output.W*output.C+cout*output.H*output.W+ (H_start)*output.W + (W_start+dil)] = ATMA[1];
	output.data[n*output.H*output.W*output.C+cout*output.H*output.W+ (H_start+dil)*output.W + (W_start)] = ATMA[2];
	output.data[n*output.H*output.W*output.C+cout*output.H*output.W+ (H_start+dil)*output.W + (W_start+dil)] = ATMA[3];
	free(dilated_input);
	free(BTx);
	free(V);
	free(M);
	free(ATM);
	free(ATMA);
}


int main(){		
	int N = 8;
	// Hin needs to be multiple of 2*dilH
	int Hin = 32;
	int Win = 32;
	int Cin = 8;
	int Cout = 8;
	int Hk = 3;
	int Wk = 3;
	int dilH = 4;
	int dilW = 4;
	// fixed to 0 for convinience
	int padH = 0;
	int padW = 0;
	// fixed to 1
	int strideH = 1;
	int strideW = 1;
	printf("N = %d\n",N);
	printf("Cout = %d\n",Cout);
	printf("Cin = %d\n",Cin);
	printf("Hin = %d\n",Hin);
	printf("Win = %d\n",Win);
	float* A = (float*)malloc(sizeof(float) * N * Hin * Win * Cin);
	float* B = (float*)malloc(sizeof(float) * Cout * Hk * Wk * Cin);
	struct tensor_ input = { .data = A, .N = N, .H = Hin, .W = Win,  .C = Cin,.SIZE = Hin * Win * N * Cin };
	struct kernel_ kernel = { .data = B, .Cout = Cout, .Cin = Cin, .H = Hk, .W = Wk, .dilH = dilH, .dilW = dilW, .padH = padH, .padW = padW, .strideH = strideH, .strideW = strideW,.SIZE = Cout * Hk * Wk* Cin};
	for (int i = 0; i <input.SIZE; i++) {
		input.data[i] = 2;
	}
	for (int i = 0; i < kernel.SIZE; i++) {
		kernel.data[i] = 2;
	}
	// golden output from CPU
	struct tensor_ golden = conv2d_direct_convolution_cpu(input,kernel);
	// print_tensor(golden);
	// preprocess input kernel and image
	struct tensor_ input_pad = tensor_pad(input, kernel.padH, kernel.padW);
	kernel.padH = 0;
	kernel.padW = 0;
	// output
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;
	// load data to gpu
	float* output_data = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	struct tensor_ output = { .data = output_data, .N = input.N,.H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	float* output_data2 = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	struct tensor_ output2 = { .data = output_data2, .N = input.N,.H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	// load data to gpu
	struct kernel_ kernel_gpu = kernel2gpu(kernel);
	struct tensor_ input_gpu = tensor2gpu(input);
	struct tensor_ output_gpu = tensor2gpu(output);
	struct tensor_ output2_gpu = tensor2gpu(output2);
	// time
	struct timespec start, stop;
	double time;


	////////////////////////////
	// Direct Convolution GPU //
	////////////////////////////

	if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}	
	// block and grid size 
	int THREAD = N*kernel.Cout*Hout*Wout;
	// dim3 dimGrid(961);
	// dim3 dimBlock(1024);
	dim3 dimGrid(THREAD);
	dim3 dimBlock(1);
	printf("Total number of thread = %d\n",  N*kernel.Cout*Hout*Wout);
	// call kernel direct_convolution_gpu
	conv2d_direct_convolution_gpu<<<dimGrid, dimBlock>>> (input_gpu,kernel_gpu,output_gpu);
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Direct convolutin GPU time is %f ns\n", time*1e9);
	// load data from device to cpu 
	output = tensor2cpu(output_gpu);
	// print_tensor(output);	
	check_tensor(golden,output);
	printf("\n\n");



	/////////////////////////////
	// Winograd Convolution GPU//
	/////////////////////////////

	// call kernel winograd
	float* U = cudaNew_(kernel.Cout*kernel.Cin*4*4);
	float* Gg = cudaNew_(kernel.Cout*kernel.Cin*4*3);
	// define Grid and Block size
	int dil = kernel.dilH;
	printf("Total number of thread = %d\n", (N*output.C * ((input.H-dil*2)/(dil*2)) *((input.W-dil*2)/(dil*2)) * dil * dil));
	dim3 dimGrid2((N*output.C * ((input.H-dil*2)/(dil*2)) *((input.W-dil*2)/(dil*2)) * dil * dil));
	dim3 dimBlock2(1);
	// Call U kernel and compute U
	wino23s1d2_GgGT_kernel<<<dimGrid2, dimBlock2>>> (kernel_gpu,Gg,U);
	// Call winograd kernel	
	// start timer
	if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}	
	wino23s1d2_BTxB_EWMM_ATMA_kernel<<< dimGrid2, dimBlock2>>> (input_gpu,output2_gpu,U,kernel.dilH);
	output2 = tensor2cpu(output2_gpu);
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Winograd GPU time is %f ns\n", time*1e9);
	// print_tensor(output2);
	check_tensor(golden,output2);
	// // free pointers
	// free_(kernel.data);
	// // free(input.data);
	// // cudaFree_(input_gpu.data);
	// // cudaFree_(kernel.data);
	// free_(output.data);
	return 0;
}	





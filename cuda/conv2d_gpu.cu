#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "utils_cu.h"
#include "im2col_cu.h"


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
	unsigned long int output_size = Hout*Wout*input.N*kernel.Cout;
	struct tensor_ output = { .data = C,.N = input.N, .H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = output_size};
	return output;
}

// empty kernel launch
__global__ void initialize(){}


// Naive implementation of direct convolution on CPU. Tensor in NCHW layout.
__global__ void conv2d_direct_convolution_gpu(struct tensor_ input, struct kernel_ kernel, struct tensor_ output){
	// Distribute computation of each output pixel to a separate cuda kernel
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long int BATCH = (output.SIZE/(blockDim.x*gridDim.x));
	unsigned long int start_idx = p * BATCH;
	for (int batch = 0; batch < BATCH;batch++){
		unsigned long int idx = start_idx + batch;
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
					unsigned long int input_idx = n * input.H * input.W * input.C + cin*input.W*input.H + hin * input.W  + win;
					unsigned long int kernel_idx = cout * kernel.Cin * kernel.H * kernel.W + cin * kernel.H * kernel.W + hk * kernel.W + wk;
					accum += (input.data[input_idx] * kernel.data[kernel_idx]);	
					
				}
			}
		}
		output.data[n * kernel.Cout * output.H * output.W + cout*output.H*output.W +  hout * output.W + wout] = accum;
	}
}


// im2col explicit GEMM conv2d with dilation. im2col imported from caffe framework.
float* conv2d_im2col_preprocessing(struct tensor_ input, struct kernel_ kernel_raw){
	struct kernel_ kernel = kernel_simple_dilation(kernel_raw);
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;
	int channels_col = input.C * kernel.H * kernel.W;
	float* im2col =(float*)malloc(sizeof(float) * input.N * Hout * Wout * channels_col);
	unsigned long int batch_size = Hout * Wout * channels_col;
	for (int n = 0; n < input.N; n++){
		float* A_n = slice(input.data,n* input.H * input.W * input.C, (n+1) * input.H * input.W * input.C);
		float* A_col_n = im2col_dilated_cpu(A_n, input.C, input.H, input.W, kernel.H, kernel.strideH, kernel.padH,kernel.dilH);
		memcpy(im2col+batch_size*n,A_col_n,sizeof(float)*batch_size);
		free_(A_n);
		free_(A_col_n);
	}	
	// print_CHW(im2col,input.N,channels_col,Hout*Wout);
	free_(kernel.data);
	return im2col;
}

__global__ void im2col_convolution_gpu(float* input, struct kernel_ kernel, struct tensor_ output, int N,int dil){
	int X = kernel.Cout;
	int Z = output.H*output.W;
    int Y = kernel.Cin*kernel.H*kernel.W;
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long int BATCH = (output.SIZE/(blockDim.x*gridDim.x));
	unsigned long int start_idx = p * BATCH;

	for (int batch = 0; batch < BATCH;batch++){
		unsigned long int idx = start_idx + batch;
		int n = idx/(X*Z);
		int x = (idx% (X*Z))/Z ;
		int z = idx%Z;

		output.data[n*X*Z+x*Z+z] = 0.0f;
		for (int cin = 0; cin < kernel.Cin; cin++) {
			for (int hk = 0; hk < kernel.H; hk=hk+dil) {
				for (int wk = 0; wk < kernel.W; wk=wk+dil) {
					int y = cin*kernel.H*kernel.W+hk*kernel.W+wk;
					output.data[n*X*Z+x*Z+z] += kernel.data[x*Y+y] * input[n*Y*Z+y*Z+z];
				}
			}
		}
	}
}



// I didn't parallelize this because offline inference doesn't need to consider GgGT computation time.
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
	
	// Distribute computation of each output tile to a separate cuda kernel
	unsigned long int p = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long int BATCH = ((input.N*output.C * ((input.H-dil*2)/(dil*2)) *((input.W-dil*2)/(dil*2)) * dil * dil))/(blockDim.x*gridDim.x);
	unsigned long int start_idx = p * BATCH;

	float dilated_input[16*4*4];
	float BTx[16*4*4];
	float V[16*4*4];
	float M[4*4];
	float ATM[4*4];
	float ATMA[4*4];

	for (unsigned long int batch = 0; batch < BATCH;batch++){
		unsigned long int idx = start_idx + batch;
		unsigned long int n = idx/(output.C * ((input.H-dil*2)/(dil*2)) *((input.W-dil*2)/(dil*2)) * dil * dil);
		int cout = (idx%(output.C*((input.H-dil*2)/(dil*2))*((input.W-dil*2)/(dil*2))*dil*dil))/(((input.H-dil*2)/(dil*2))*((input.W-dil*2)/(dil*2))*dil*dil);
		int h = (idx%(((input.H-dil*2)/(dil*2))*((input.W-dil*2)/(dil*2))*dil*dil))/(((input.W-dil*2)/(dil*2))*dil*dil)*(dil*2);
		int w = ((idx%(((input.W-dil*2)/(dil*2))*dil*dil))/(dil*dil))*(dil*2);
		int hh = (idx%(dil*dil))/(dil);
		int ww = idx%dil;

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

		// Use NCHW data layout for inner product
		// intialize M
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				M[i*4+j] = 0;
			}
		}

		// calculate M 
		for (int cin = 0; cin < input.C; ++cin){
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					M[i*4+j] += U[cout*input.C*16+cin*16+i*4+j] * V[cin*16+i*4+j];
				}
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

		// map ATMA to the output matrix:
		int H_start = h+hh;
		int W_start = w+ww;
		output.data[n*output.H*output.W*output.C+cout*output.H*output.W + H_start*output.W + W_start] = ATMA[0];
		output.data[n*output.H*output.W*output.C+cout*output.H*output.W+ (H_start)*output.W + (W_start+dil)] = ATMA[1];
		output.data[n*output.H*output.W*output.C+cout*output.H*output.W+ (H_start+dil)*output.W + (W_start)] = ATMA[2];
		output.data[n*output.H*output.W*output.C+cout*output.H*output.W+ (H_start+dil)*output.W + (W_start+dil)] = ATMA[3];
	}
}


int main(){		
	int N = 16;
	// Hin needs to be multiple of 2*dilH
	int Hin = 28;
	int Win = 28;
	int Cin = 16;
	int Cout = 16;
	int Hk = 3;
	int Wk = 3;
	int dilH = 2;
	int dilW = 2;
	// fixed to 0 for convinience
	int padH = 0;
	int padW = 0;
	// fixed to 1
	int strideH = 1;
	int strideW = 1;
	// Choose from 1 to 1024
	int p = 32;
	// Choose from 1 to 1024
	int block_size = 32;
	int dil = dilH;


	printf("N = %d\n",N);
	printf("Cout = %d\n",Cout);
	printf("Cin = %d\n",Cin);
	printf("Hin = %d\n",Hin);
	printf("Win = %d\n",Win);
	printf("dil = %d\n",dil);

	

	////////////////////
	// Initialization //
	////////////////////

	float* A = (float*)malloc(sizeof(float) * N * Hin * Win * Cin);
	float* B = (float*)malloc(sizeof(float) * Cout * Hk * Wk * Cin);
	unsigned long int input_size = Hin * Win * N * Cin;
	unsigned long int kernel_size = Cout * Hk * Wk* Cin;
	struct tensor_ input = { .data = A, .N = N, .H = Hin, .W = Win,  .C = Cin,.SIZE =  input_size };
	struct kernel_ kernel = { .data = B, .Cout = Cout, .Cin = Cin, .H = Hk, .W = Wk, .dilH = dilH, .dilW = dilW, .padH = padH, .padW = padW, .strideH = strideH, .strideW = strideW,.SIZE = kernel_size};
	for (int i = 0; i <input.SIZE; i++) {
		input.data[i] = 2;
	}
	for (int i = 0; i < kernel.SIZE; i++) {
		kernel.data[i] = 2;
	}
	// Golden output: disenable when matrix size is large, because it takes long time to compute
	// struct tensor_ golden = conv2d_direct_convolution_cpu(input,kernel);
	// print_tensor(golden);


	///////////////////
	// preprocessing //
	///////////////////
	struct tensor_ input_pad = tensor_pad(input, kernel.padH, kernel.padW);
	kernel.padH = 0;
	kernel.padW = 0;
	// output
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;
	unsigned long int output_size = Hout*Wout*input.N*kernel.Cout;

	// direct convolution
	float* output_data = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	struct tensor_ output = { .data = output_data, .N = input.N,.H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = output_size};
	// winograd
	float* output_data2 = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	struct tensor_ output2 = { .data = output_data2, .N = input.N,.H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = output_size};
	// im2col
	float* output_data3 = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	struct tensor_ output3 = { .data = output_data3, .N = input.N,.H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = output_size};
	
	

	// // Im2col intialization
	// // disabled, because im2col transformation is only available in CPU.
	struct kernel_ dilated_kernel = kernel_simple_dilation(kernel);
	unsigned long int im2col_size = input.N*input.C * dilated_kernel.H * dilated_kernel.W * Hout * Wout;
	float* im2col_input = conv2d_im2col_preprocessing(input,kernel);
	float* im2col_input_gpu = data2gpu(im2col_input,im2col_size);
	struct kernel_ dilated_kernel_gpu = kernel2gpu(dilated_kernel);


	// load data to gpu
	struct kernel_ kernel_gpu = kernel2gpu(kernel);
	struct tensor_ input_gpu = tensor2gpu(input);
	struct tensor_ output_gpu = tensor2gpu(output);
	struct tensor_ output2_gpu = tensor2gpu(output2);
	struct tensor_ output3_gpu = tensor2gpu(output3);
	// time
	struct timespec start, stop;
	double time;

	
	unsigned long int THREAD = (N*output.C * ((input.H-dil*2)/(dil*2)) *((input.W-dil*2)/(dil*2)) * dil * dil);
	printf("Grid Size = %d\n",  MAX(p/block_size,1));
	printf("Block Size = %d\n",  MIN(block_size,THREAD));
	printf("Output size = %lu\n\n",  output_size);
	dim3 dimGrid(MAX(p/block_size,1));
	dim3 dimBlock(MIN(block_size,THREAD));
	


	// First time kernel launch takes long time. So start an empty one first.
	initialize<<<dimGrid, dimBlock>>> ();

	////////////////////////////
	// Direct Convolution GPU //
	////////////////////////////
	if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}	
	// call kernel direct_convolution_gpu
	conv2d_direct_convolution_gpu<<<dimGrid, dimBlock>>> (input_gpu,kernel_gpu,output_gpu);
	cudaDeviceSynchronize();
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Direct convolutin GPU time is %f ns\n", time*1e9);
	// load data from device to cpu 
	output = tensor2cpu(output_gpu);
	printf("output[100] = %.f\n",output.data[100]);
	// print_tensor(output);	
	// check_tensor(golden,output);
	printf("\n\n");


	////////////////////////////
	// Im2col Convolution GPU //
	////////////////////////////
	if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}	
	// call kernel direct_convolution_gpu
	im2col_convolution_gpu<<<dimGrid, dimBlock>>> (im2col_input_gpu,dilated_kernel_gpu,output3_gpu,input.N,dil);
	cudaDeviceSynchronize();
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Im2col convolution GPU time is %f ns\n", time*1e9);
	// load data from device to cpu 
	output3 = tensor2cpu(output3_gpu);
	printf("output[100] = %.f\n",output3.data[100]);
	// print_tensor(output3);	
	// check_tensor(golden,output3);
	printf("\n\n");


	
	/////////////////////////////
	// Winograd Convolution GPU//
	/////////////////////////////
	// call kernel winograd
	float* U = cudaNew_(kernel.Cout*kernel.Cin*4*4);
	float* Gg = cudaNew_(kernel.Cout*kernel.Cin*4*3);
	// Call U kernel and compute U
	wino23s1d2_GgGT_kernel<<<1,1>>> (kernel_gpu,Gg,U);
	// Call winograd kernel	
	// start timer

	if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}	
	wino23s1d2_BTxB_EWMM_ATMA_kernel<<< dimGrid, dimBlock>>> (input_gpu,output2_gpu,U,kernel.dilH);
	cudaDeviceSynchronize();
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Winograd GPU time is %f ns\n", time*1e9);
	output2 = tensor2cpu(output2_gpu);
	// print_tensor(output2);
	printf("output[100] = %.f\n",output2.data[100]);
	check_tensor(output,output2);


	// // free pointers
	cudaFree_(kernel.data);
	cudaFree_(input_gpu.data);
	cudaFree_(kernel.data);
	// free_(golden.data);
	free_(output.data);
	free_(output2.data);
	free_(output3.data);
	return 0;
}	





#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils_cu.h"
#include <string.h>
// #include <winograd_transform.h>


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
						for (int hk = 0; hk < kernel.H;hk=hk+2){
							for (int wk = 0; wk < kernel.W;wk=wk+2){	
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
	float accum = 0.0f;
	for (int cin = 0; cin < input.C;cin++){
		for (int hk = 0; hk < kernel.H;hk=hk+2){
			for (int wk = 0; wk < kernel.W;wk=wk+2){	
				int hin = (hout * kernel.strideH + hk)-kernel.padH;
				int win = (wout * kernel.strideW + wk)-kernel.padW;
				int input_idx = n * input.H * input.W * input.C + cin*input.W*input.H + hin * input.W  + win;
				int kernel_idx = cout * kernel.Cin * kernel.H * kernel.W + cin * kernel.H * kernel.W + hk * kernel.W + wk;
				accum += (input.data[input_idx] * kernel.data[kernel_idx]);	
				
			}
		}
	}
	output.data[n * kernel.Cout * output.H * output.W + cout*output.H*output.W +  hout * output.W + wout] = accum;
}


int main(){		
	int N = 4;
	int Hin = 128;
	int Win = 128;
	int Cin = 16;
	int Cout = 16;
	int Hk = 3;
	int Wk = 3;
	int dilH = 2;
	int dilW = 2;
	int padH = 0;
	int padW = 0;
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
	struct tensor_ output2 = conv2d_direct_convolution_cpu(input,kernel);
	// print_tensor(output2);
	struct kernel_ kernel_ = kernel_simple_dilation(kernel);
	
	// output
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;

	// load data to gpu
	float* output_data = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	struct tensor_ output = { .data = output_data, .N = input.N,.H = Hout, .W = Wout,  .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	struct kernel_ kernel_gpu = kernel2gpu(kernel_);
	struct tensor_ input_gpu = tensor2gpu(input);
	struct tensor_ output_gpu = tensor2gpu(output);
	struct timespec start, stop;
	double time;

	if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}	

	int THREAD = N*kernel.Cout*Hout*Wout;
	dim3 dimGrid(961);
	dim3 dimBlock(1024);
	conv2d_direct_convolution_gpu<<< dimGrid, dimBlock>>> (input_gpu,kernel_gpu,output_gpu);

	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("time is %f ns\n", time*1e9);

	output = tensor2cpu(output_gpu);
	// print_tensor(output);	
	check_tensor(output, output2);





	free_(kernel.data);
	// free(input.data);
	cudaFree_(input_gpu.data);
	cudaFree_(kernel_.data);
	free_(output.data);
	
	return 0;
}	





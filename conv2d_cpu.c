#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <conv.h>
#include <im2col.h>
#include <utils.h>


// Naive implementation of convolution using sliding window on CPU. Tensor in NCHW layout.
struct tensor_ conv2d_implicit_GEMM_cpu(struct tensor_ input, struct kernel_ kernel_raw){
	struct kernel_ kernel = kernel_simple_dilation(kernel_raw);
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;
	float accum;
	int input_idx, kernel_idx,hin,win;
	float* C = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	for (int n = 0; n < input.N;n++){
		for (int cout = 0; cout < kernel.Cout; cout++){
			for (int hout = 0; hout < Hout; hout++){
				for (int wout = 0; wout < Wout; wout++){
					accum = 0.0;
					for (int cin = 0; cin < input.C;cin++){
						for (int hk = 0; hk < kernel.H;hk++){
							for (int wk = 0; wk < kernel.W;wk++){	
								hin = (hout * kernel.strideH + hk)-kernel.padH;
								win = (wout * kernel.strideW + wk)-kernel.padW;
								if (hin < 0 || hin >= input.H || win < 0 || win >= input.W) {
									accum += 0;
								}
								else {
									input_idx = find_tensor_idx(n, cin, hin, win, input);
									kernel_idx = find_kernel_idx(cout, cin, hk, wk, kernel);
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
	struct tensor_ output = { .data = C, .H = Hout, .W = Wout, .N = input.N, .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	return output;
}



// im2col explicit GEMM conv2d with dilation. im2col imported from caffe framework.
struct tensor_ conv2d_im2col_GEMM_cpu(struct tensor_ input, struct kernel_ kernel_raw){
	struct kernel_ kernel = kernel_simple_dilation(kernel_raw);
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;

	float* C =(float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);


	for (int n = 0; n < input.N;n++){
		float* A_n = slice(input.data,n* input.H * input.W * input.C, (n+1) * input.H * input.W * input.C);
		float* A_col_n = im2col_dilated_cpu(A_n, input.C, input.H, input.W, kernel.H, kernel.strideH, kernel.padH,kernel.dilH);
		int dilate_ksize = (kernel.dilH - 1) * (kernel.H + 1) + kernel.W;
		int channels_col = input.C * kernel.H * kernel.W;
		float* B = kernel.data;
		int x = kernel.Cout;
		int y = channels_col;
		int z = Hout*Wout;	
		float* temp = im2col_mm(B,A_col_n,x,y,z);
		//print_W(temp, x * z);
		int batch = x*z;
		memcpy(C+batch*n,temp,sizeof(float)*batch);
		//print_W(C, batch);
	}	
	struct tensor_ output = { .data = C, .H = Hout, .W = Wout, .N = input.N, .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	return output;
}


int main(){		

	int N = 2;
	int Hin = 15;
	int Win = 15;
	int Cin = 2;
	int Cout = 2;
	int Hk = 5;
	int Wk = 5;
	int dilH = 2;
	int dilW = 2;
	int padH = 2;
	int padW = 2;
	int strideH = 2;
	int strideW = 2;
	float* A = (float*)malloc(sizeof(float) * N * Hin * Win * Cin);
	float* B = (float*)malloc(sizeof(float) * Cout * Hk * Wk * Cin);
	struct tensor_ input = { .data = A, .H = Hin, .W = Win, .N = N, .C = Cin,.SIZE = Hin * Win * N * Cin };
	struct kernel_ kernel = { .data = B, .Cout = Cout, .Cin = Cin, .H = Hk, .W = Wk, .dilH = dilH, .dilW = dilW, .padH = padH, .padW = padW, .strideH = strideH, .strideW = strideW,.SIZE = Cout * Hk * Wk* Cin };

	for (int i = 0; i <input.SIZE; i++) {
		input.data[i] = i;
	}
	for (int i = 0; i < kernel.SIZE; i++) {
		kernel.data[i] = 3;
	}

	//struct timespec start, stop;
	//double time;
	//if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

	//print_kernel(kernel);
	struct tensor_ output = conv2d_implicit_GEMM_cpu(input,kernel);
	print_tensor(output);

	output = conv2d_im2col_GEMM_cpu(input, kernel);
	print_tensor(output);


	//if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	//time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	//printf("time is %f ns\n", time*1e9);

	//free(input.data);
	//free(kernel.data);
	//free(output.data);
	return 0;
}	





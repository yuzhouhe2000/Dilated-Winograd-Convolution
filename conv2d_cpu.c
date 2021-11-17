#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <conv.h>
#include <im2col.h>
#include <utils.h>
#include <string.h>
//#include <winograd_transform.h>

// Naive implementation of direct convolution on CPU. Tensor in NCHW layout.
struct tensor_ conv2d_direct_convolution_cpu(struct tensor_ input, struct kernel_ kernel_raw){
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
	free(kernel.data);
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
		//print_CHW(A_n, input.C, input.H, input.W);
		float* A_col_n = im2col_dilated_cpu(A_n, input.C, input.H, input.W, kernel.H, kernel.strideH, kernel.padH,kernel.dilH);
		int dilate_ksize = (kernel.dilH - 1) * (kernel.H + 1) + kernel.W;
		int channels_col = input.C * kernel.H * kernel.W;
		int x = kernel.Cout;
		int y = channels_col;
		int z = Hout*Wout;	
		//print_CHW(A_col_n, channels_col, Hout,Wout);
		float* temp = im2col_mm(kernel.data,A_col_n,x,y,z);
		int batch = x*z;
		memcpy(C+batch*n,temp,sizeof(float)*batch);
		//print_W(C, batch);
		free_(A_n);
		free_(A_col_n);
		free_(temp);
	}	
	free_(kernel.data);
	struct tensor_ output = { .data = C, .H = Hout, .W = Wout, .N = input.N, .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	return output;
}


// F(2,3) dilated convolution, dilation = 2, stride = 1. Tensor in NCHW layout.
// Input matrix breaks into 8x8 tiles ((m+r-1)*2 = 8), each consist of four 7x7 sparse tiles (converted to 4x4 winograd input).
// Output matrix breaks into 4x4 tiles, each consist of four 4x4 sparse tiles (converted to 2x2 winograd output).
// Overlap by r-1 = 2, because dilation =2 -> overlap = 4

struct tensor_ conv2d_dilated_winograd23s1d2_cpu1(struct tensor_ input_raw, struct kernel_ kernel){
	// TODO: use better way to pad
	struct tensor_ input = tensor_pad(input_raw, kernel.padH, kernel.padW);
	kernel.padH = 0;
	kernel.padW = 0;
	float tile;
	int posH,posW,A_n_idx,tile_idx,C_idx;
	int Hout = ((input.H + 2*kernel.padH - kernel.dilH * (kernel.H - 1) - 1)/kernel.strideH) + 1;
	int Wout = ((input.W + 2*kernel.padW - kernel.dilW * (kernel.W - 1) - 1)/kernel.strideW) + 1;
	float* C = (float*)malloc(sizeof(float) * input.N * Hout * Wout * kernel.Cout);
	// 3x3 kernel dilation -> 5x5
	struct kernel_ dilated_kernel = kernel_simple_dilation(kernel);
	

	// For each batch
	for (int n = 0; n < input.N;n++){
		float* A_n = slice(input.data,n* input.H * input.W * input.C, (n+1) * input.H * input.W * input.C);
		// For each tile group (4 tiles)
		// Overlap = 4		
		for (int hin = 0; hin < input.H-7; hin=hin+4){
			for (int win = 0; win < input.W-7; win=win+4){
				// Initialize tile group (4 tiles)
				// TODO: maybe NHWC better? Transpose?
				float* tile_group = (float*)malloc(sizeof(float) * input.C * 8*8);
				for (int cin = 0; cin < input.C; cin++){		
					for (int yy = 0; yy < 8; yy++){
						for (int xx = 0; xx < 8; xx++){
							posH = hin+yy;
							posW = win+xx;
							A_n_idx = cin*input.H*input.W+posH*input.W+posW;
							tile_group[cin*64+yy*8+xx] = A_n[A_n_idx];
						}
					}
				}
				//print_CHW(tile_group, kernel.Cin, 8, 8);
				// winograd on tile
				// output matrix size = 4x4, input matrix size = 8x8, kernel size = 5x5
				float* tile_output = tile_wino23s1d2_cpu(tile_group,dilated_kernel,Hout,Wout);
				free_(tile_group);
				// memcpy tile result to output matrix C
				for (int cout = 0; cout < kernel.Cout; cout++){		
					for (int yy = 0; yy < 4; yy++){
						for (int xx = 0; xx < 4; xx++){
							posH = hin+yy;
							posW = win+xx;
							tile_idx = cout * 16 + yy * 4 + xx;
							C_idx = find_NCHW_idx(n, cout, posH, posW, input.N, kernel.Cout, Hout, Wout);
							C[C_idx] = tile_output[tile_idx];
						}
					}
				}
				free_(tile_output);
			}
		}	
		free_(A_n);
	}	
	struct tensor_ output = { .data = C, .H = Hout, .W = Wout, .N = input.N, .C = kernel.Cout,.SIZE = Hout*Wout*input.N*kernel.Cout};
	return output;
}



int main(){		

	int N = 1;
	int Hin = 12;
	int Win = 12;
	int Cin = 1;
	int Cout = 1;
	int Hk = 3;
	int Wk = 3;
	int dilH = 2;
	int dilW = 2;
	int padH = 0;
	int padW = 0;
	int strideH = 1;
	int strideW = 1;
	float* A = (float*)malloc(sizeof(float) * N * Hin * Win * Cin);
	float* B = (float*)malloc(sizeof(float) * Cout * Hk * Wk * Cin);
	struct tensor_ input = { .data = A, .H = Hin, .W = Win, .N = N, .C = Cin,.SIZE = Hin * Win * N * Cin };
	struct kernel_ kernel = { .data = B, .Cout = Cout, .Cin = Cin, .H = Hk, .W = Wk, .dilH = dilH, .dilW = dilW, .padH = padH, .padW = padW, .strideH = strideH, .strideW = strideW,.SIZE = Cout * Hk * Wk* Cin};

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
	struct tensor_ output = conv2d_direct_convolution_cpu(input,kernel);
	print_tensor(output);

	output = conv2d_im2col_GEMM_cpu(input, kernel);
	print_tensor(output);

	output = conv2d_dilated_winograd23s1d2_cpu1(input, kernel);
	print_tensor(output);


	//if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	 
	//time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	//printf("time is %f ns\n", time*1e9);

	free_(input.data);
	free_(kernel.data);
	free_(output.data);
	return 0;
}	




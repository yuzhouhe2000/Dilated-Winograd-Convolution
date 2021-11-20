#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"
#include "winograd_transform.h"
#include <omp.h>

// Define winograd transformation matrix.
const float wino23s1d2_BT[4][7] = { {1.0f,0.0f,0.0f,0.0f,-1.0f,0.0f,0.0f},
								   {0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f},
								   {0.0f,0.0f,-1.0f,0.0f,1.0f,0.0f,0.0f},
								   {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,-1.0f} };

const float wino23s1d2_G[4][5] = { {1.0f,0.0f,0.0f,0.0f,0.0f},
								 {1.0f / 2,0.0f,1.0f / 2,0.0f,1.0f / 2},
								 {1.0f / 2,0.0f,-1.0f / 2,0.0f,1.0f / 2},
								 {0.0f,0.0f,0.0f,0.0f,1.0f} };

const float wino23s1d2_AT[3][4] = { {1.0f,1.0f,1.0f,0.0f},
								   {0.0f,0.0f,0.0f,0.0f},
								   {0.0f,1.0f,-1.0f,-1.0f} };


// TODO: organize code
float* wino23s1d2_GgGT_cpu(struct kernel_ kernel,float* Gg){
	float accum;
	int kernel_idx, Gg_idx;
	// TODO: faster MM  
	// (4x5),(5x5) -> (4,5)
	for (int cout = 0; cout < kernel.Cout; ++cout){
		for (int cin = 0; cin < kernel.Cin; ++cin){
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 5; j=j+2) {
					accum = 0.0f;
					for (int k = 0; k < 5; k = k + 2) {
						kernel_idx = find_kernel_idx(cout, cin, k, j, kernel);
						accum += wino23s1d2_G[i][k] * kernel.data[kernel_idx];
					}
					Gg_idx = find_CCHW_idx(cout, cin, i, j, kernel.Cout, kernel.Cin, 4, 5);
					Gg[Gg_idx] = accum;
				}
			}
		}
	}
	float* GgGT = (float*)malloc(sizeof(float)*kernel.Cout*kernel.Cin*4*4);
	// (4x5),(5x4) -> (4,4)
	for (int cout = 0; cout < kernel.Cout; ++cout){
		for (int cin = 0; cin < kernel.Cin; ++cin){
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					accum = 0.0f;
					for (int k = 0; k < 5; k=k+2) {
						Gg_idx = find_CCHW_idx(cout, cin, i, k, kernel.Cout, kernel.Cin, 4, 5);
						accum += Gg[Gg_idx] * wino23s1d2_G[j][k];
					}
					GgGT[cout*kernel.Cin*16+cin*16+i*4+j] = accum;
				}
			}
		}
	}
	return GgGT;
}

float* wino23s1d2_BTxB_cpu(float* input_partition,int Cin,float* BTx){
	float accum;
	
	// TODO: faster MM  
	// (4x7),(7x7) -> (4,7)
	for (int cin = 0; cin < Cin; ++cin){
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 7; j=j+2) {
				accum = 0.0f;
				for (int k = 0; k < 7; k=k+2) {
					accum += wino23s1d2_BT[i][k] * input_partition[cin*49+k*7+j];
				}
				BTx[cin *28 + i * 7 + j] = accum;
			}
		}
	}

	float* BTxB = (float*)malloc(sizeof(float)*Cin*4*4);
	// (4x7),(7x4) -> (4,4)
	for (int cin = 0; cin < Cin; ++cin){
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				accum = 0.0f;
				for (int k = 0; k < 7; k=k+2) {
					accum += BTx[cin * 28 + i * 7 + k] * wino23s1d2_BT[j][k];
				}
				BTxB[cin*16+i*4+j] = accum;
			}
		}
	}
	return BTxB;
}

float* elementwise_mm_NHWC_ver_cpu(float* U, float* V, int Cout, int Cin) {
	float* UT = NCHW_2_NHWC(U,Cout,Cin,4,4);
	float* VT = NCHW_2_NHWC(V,1,Cin,4,4);

	float accum;
	float* M = (float*)malloc(sizeof(float) * Cout * 4 * 4);
	// TODO: parallize element wise multiplication
	for (int cout = 0; cout < Cout; ++cout) {
		for (int h = 0; h < 4; ++h) {
			for (int w = 0; w < 4; ++w) {
				accum = 0.0f;
				// TODO: maybe HWC instead of CHW? or change algorithm?
				for (int cin = 0; cin < Cin; ++cin) {
					accum += UT[find_CCHW_idx(cout,h,w,cin,Cout,4,4,Cin)] * VT[find_CCHW_idx(0, h, w, cin, 1, 4, 4, Cin)];
				}
				M[cout * 16 + h * 4 + w] = accum;
			}
		}
	}
	free_(U);
	free_(V);
	free_(UT);
	free(VT);
	return M;
}


float* elementwise_mm_cpu(float* U,float* V,int Cout,int Cin){
	float accum;
	float* M = (float*)malloc(sizeof(float)*Cout*4*4);
	// TODO: parallize element wise multiplication
	for (int cout = 0; cout < Cout; ++cout){
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				accum = 0.0f;
				// TODO: maybe HWC instead of CHW? or change algorithm?
				for (int cin = 0; cin < Cin; ++cin){
					accum += U[cout*Cin*16+cin*16+i*4+j] * V[cin*16+i*4+j];
				}
				M[cout*16+i*4+j] = accum;
			}
		}
	}
	free_(U);
	free_(V);
	return M;
}

float* wino23s1d2_ATMA_cpu(float* M,int Cout,float* ATM){
	float accum;
	
	// TODO: faster MM  
	// (3x4),(4x4) -> (3,4)
	for (int cout = 0; cout < Cout; ++cout){
		for (int i = 0; i < 3; i=i+2) {
			for (int j = 0; j < 4; ++j) {
				accum = 0.0f;
				for (int k = 0; k < 4; ++k) {
					accum += wino23s1d2_AT[i][k] * M[cout*16+k*4+j];
				}
				ATM[cout*12+i*4+j] = accum;
			}
		}
	}
	
	float* ATMA = (float*)malloc(sizeof(float)*Cout*3*3);
	// (3x4),(4x3) -> (3,3)
	for (int cout = 0; cout < Cout; ++cout){
		for (int i = 0; i < 3; i = i+2) {
			for (int j = 0; j < 3; j =j+2) {
				accum = 0.0f;
				for (int k = 0; k < 4; ++k) {
					accum += ATM[cout*12+i*4+k] * wino23s1d2_AT[j][k];
				}
				ATMA[cout*9+i*3+j] = accum;
			}
		}
	}
	return ATMA;
}

float* tile_wino23s1d2_cpu(float* tile_group,struct kernel_ kernel,int Hout,int Wout){
	int Cout = kernel.Cout;
	int Cin = kernel.Cin;
	float* merged_tile = (float*)malloc(sizeof(float)*Cout*4*4);
	//print_CHW(tile_group, kernel.Cin,8,8);
	float* input_partition = (float*)malloc(sizeof(float) * Cin * 7 * 7);
	float* Gg = (float*)malloc(sizeof(float) * (kernel.Cout * kernel.Cin * 4 * 5));
	float* BTx = (float*)malloc(sizeof(float) * Cin * 4 * 7);
	float* ATM = (float*)malloc(sizeof(float) * Cout * 3 * 4);

	/*omp_set_num_threads(4); */
	int tile_idx;
	#pragma omp parallel for
	for(tile_idx = 0;tile_idx<4;++tile_idx){
		// TODO: parallel partition
		// Divide into four partitions
		int xadd = 0;
		int yadd = 0;
		// tile_idx = 2 or 3 
		if (tile_idx >= 2){
			yadd = 1;
		}
		// tile_idx = 1 or 3 
		if (tile_idx%2 == 1){
			xadd = 1;
		}
		for (int cin = 0; cin < Cin; ++cin){
			for (int yy = 0; yy < 7; ++yy){
				for (int xx = 0; xx < 7; ++xx){
					input_partition[cin*49+yy*7+xx] = tile_group[cin*64+(yy+yadd)*8+(xx+xadd)];
				}
			}
		}

		//print_CHW(input_partition,Cin,7,7);
		//perform winograd: stride=1, dilation=2, F(2x2,3x3)
		float* U = wino23s1d2_GgGT_cpu(kernel,Gg);
		//printf("U:\n");
		//print_CHW(U, Cout * Cin, 4, 4);
		float* V = wino23s1d2_BTxB_cpu(input_partition,Cin,BTx);
		//printf("V:\n");
		//print_CHW(BTx, Cin, 4, 7);
		//print_CHW(V,Cin, 4, 4);
		float* M = elementwise_mm_cpu(U,V,Cout,Cin);
		// NHWC is Slower because of transpose here
		//float* M = elementwise_mm_NHWC_ver_cpu(U, V, Cout, Cin);
		//printf("M:\n");
		//print_CHW(M, Cout, 4, 4);
		float* tile_partition = wino23s1d2_ATMA_cpu(M,Cout,ATM);
		//printf("tile:\n");
		//print_CHW(tile_partition, Cout, 3, 3);
		
		// TODO: parallel merge
		 //Merge into single output
		for (int cout = 0; cout < Cout; ++cout){
			if (tile_idx == 0){
				merged_tile[cout*16+0] = tile_partition[cout*9+0];
				merged_tile[cout*16+2] = tile_partition[cout*9+2];
				merged_tile[cout*16+8] = tile_partition[cout*9+6];
				merged_tile[cout*16+10] = tile_partition[cout*9+8];
			}
			else if (tile_idx == 1){
				merged_tile[cout*16+1] = tile_partition[cout*9+0];
				merged_tile[cout*16+3] = tile_partition[cout*9+2];
				merged_tile[cout*16+9] = tile_partition[cout*9+6];
				merged_tile[cout*16+11] = tile_partition[cout*9+8];
			}
			else if (tile_idx == 2){
				merged_tile[cout*16+4] = tile_partition[cout*9+0];
				merged_tile[cout*16+6] = tile_partition[cout*9+2];
				merged_tile[cout*16+12] = tile_partition[cout*9+6];
				merged_tile[cout*16+14] = tile_partition[cout*9+8];
			}
			else{
				merged_tile[cout*16+5] = tile_partition[cout*9+0];
				merged_tile[cout*16+7] = tile_partition[cout*9+2];
				merged_tile[cout*16+13] = tile_partition[cout*9+6];
				merged_tile[cout*16+15] = tile_partition[cout*9+8];
			}
		}
		free_(M);
		free_(tile_partition);
	}
	
	free_(Gg);
	free_(BTx);
	free_(ATM);
	free_(input_partition);
	return merged_tile;
}




#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"
#include "winograd_transform.h"
#include <omp.h>

// GgGT is bottleneck when Cout and Cin is large
float* wino23s1d2_GgGT_cpu(struct kernel_ kernel,float* Gg){
	// (4x5),(5x5) -> (4,5)
	// Sparse
	int cout;
	#pragma omp parallel for
	for (cout = 0; cout < kernel.Cout; ++cout){
		for (int cin = 0; cin < kernel.Cin; ++cin){
			for (int j = 0; j < 5; j=j+2) {
				Gg[cout*kernel.Cin*4*5+cin*4*5+0*5+j] = kernel.data[cout*kernel.Cin*5*5+cin*5*5+0*5+j];
				Gg[cout*kernel.Cin*4*5+cin*4*5+1*5+j] = 0.5f*kernel.data[cout*kernel.Cin*5*5+cin*5*5+0*5+j]+
														0.5f*kernel.data[cout*kernel.Cin*5*5+cin*5*5+2*5+j]+
														0.5f*kernel.data[cout*kernel.Cin*5*5+cin*5*5+4*5+j];
				Gg[cout*kernel.Cin*4*5+cin*4*5+2*5+j] = 0.5f*kernel.data[cout*kernel.Cin*5*5+cin*5*5+0*5+j]-
														0.5f*kernel.data[cout*kernel.Cin*5*5+cin*5*5+2*5+j]+
														0.5f*kernel.data[cout*kernel.Cin*5*5+cin*5*5+4*5+j];
				Gg[cout*kernel.Cin*4*5+cin*4*5+3*5+j] = kernel.data[cout*kernel.Cin*5*5+cin*5*5+4*5+j];
			}
		}
	}

	float* GgGT = (float*)malloc(sizeof(float)*kernel.Cout*kernel.Cin*4*4);
	// (4x5),(5x4) -> (4,4)
	#pragma omp parallel for
	for (cout = 0; cout < kernel.Cout; ++cout){
		for (int cin = 0; cin < kernel.Cin; ++cin){
			for (int i = 0; i < 4; ++i) {
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+0] = Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+0];
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+1] = 0.5f*Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+0]+
														0.5f*Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+2]+
														0.5f*Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+4];
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+2] = 0.5f*Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+0]-
														0.5f*Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+2]+
														0.5f*Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+4];
				GgGT[cout*kernel.Cin*4*4+cin*4*4+i*4+3] = Gg[cout*kernel.Cin*4*5+cin*4*5+i*5+4];
			}
		}
	}
	return GgGT;
}

float* wino23s1d2_BTxB_cpu(float* input_partition,int Cin,float* BTx){
	float accum;
	// Sparse
	for (int cin = 0; cin < Cin; ++cin){
		for (int j = 0; j < 7; j=j+2) {
			// when k = 0,2,4,6
			BTx[cin * 28 + 0 * 7 + j] = input_partition[cin*49+0*7+j] - input_partition[cin*49+4*7+j];
			BTx[cin * 28 + 1 * 7 + j] = input_partition[cin*49+2*7+j] + input_partition[cin*49+4*7+j];
			BTx[cin * 28 + 2 * 7 + j] = - input_partition[cin*49+2*7+j] + input_partition[cin*49+4*7+j];
			BTx[cin * 28 + 3 * 7 + j] = input_partition[cin*49+2*7+j] - input_partition[cin*49+6*7+j];
		}
	}
	float* BTxB = (float*)malloc(sizeof(float)*Cin*4*4);
	// (4x7),(7x4) -> (4,4)
	for (int cin = 0; cin < Cin; ++cin){
		for (int i = 0; i < 4; ++i) {
			BTxB[cin * 16 + i * 4 + 0] = BTx[cin*28+i*7+0] - BTx[cin*28+i*7+4];
			BTxB[cin * 16 + i * 4 + 1] = BTx[cin*28+i*7+2] + BTx[cin*28+i*7+4];
			BTxB[cin * 16 + i * 4 + 2] = - BTx[cin*28+i*7+2] + BTx[cin*28+i*7+4];
			BTxB[cin * 16 + i * 4 + 3] = BTx[cin*28+i*7+2] - BTx[cin*28+i*7+6];
		}
	}
	return BTxB;
}


float* wino23s1d2_ATMA_cpu(float* M,int Cout,float* ATM){
	// (3x4),(4x4) -> (3,4)
	for (int cout = 0; cout < Cout; ++cout){
		for (int j = 0; j < 4; j++) {
			ATM[cout*12+0*4+j] = M[cout*16+0*4+j]+M[cout*16+1*4+j]+M[cout*16+2*4+j];
			ATM[cout*12+2*4+j] = M[cout*16+1*4+j]-M[cout*16+2*4+j]-M[cout*16+3*4+j];

		}
	}
	float* ATMA = (float*)malloc(sizeof(float)*Cout*3*3);
	// (3x4),(4x3) -> (3,2)
	for (int cout = 0; cout < Cout; ++cout){
		for (int i = 0; i < 3; i=i+2) {
			ATMA[cout * 9 + i * 3 + 0] = ATM[cout*12+i*4+0]+ATM[cout*12+i*4+1]+ATM[cout*12+i*4+2];
			ATMA[cout * 9 + i * 3 + 2] = ATM[cout*12+i*4+1]-ATM[cout*12+i*4+2]-ATM[cout*12+i*4+3];
		}
	}
	return ATMA;
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
	// free_(U);
	free_(V);
	return M;
}

float* tile_wino23s1d2_cpu(float* tile_group,struct kernel_ kernel,int Hout,int Wout,float* U){
	int Cout = kernel.Cout;
	int Cin = kernel.Cin;
	float* merged_tile = (float*)malloc(sizeof(float)*Cout*4*4);
	//print_CHW(tile_group, kernel.Cin,8,8);
	float* input_partition = (float*)malloc(sizeof(float) * Cin * 7 * 7);
	
	float* BTx = (float*)malloc(sizeof(float) * Cin * 4 * 7);
	float* ATM = (float*)malloc(sizeof(float) * Cout * 3 * 4);

	/*omp_set_num_threads(4); */
	int tile_idx;
	// #pragma omp parallel for
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
			for (int yy = 0; yy < 7; yy=yy+2){
				for (int xx = 0; xx < 7; xx = xx+2){
					input_partition[cin*49+yy*7+xx] = tile_group[cin*64+(yy+yadd)*8+(xx+xadd)];
				}
			}
		}

		struct timespec start, stop;
		double time;

		//perform winograd: stride=1, dilation=2, F(2x2,3x3)
		// float* Gg = (float*)malloc(sizeof(float) * (kernel.Cout * kernel.Cin * 4 * 5));
		// float* U = wino23s1d2_GgGT_cpu(kernel,Gg);
		float* V = wino23s1d2_BTxB_cpu(input_partition,Cin,BTx);
		float* M = elementwise_mm_cpu(U,V,Cout,Cin);
		float* tile_partition = wino23s1d2_ATMA_cpu(M,Cout,ATM);

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
	
	free_(BTx);
	free_(ATM);
	free_(input_partition);
	return merged_tile;
}




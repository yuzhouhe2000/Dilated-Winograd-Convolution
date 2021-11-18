// Winograd F(, 5x5), 8x8 input tiles
float* wino23s1d2_GgGT_cpu(struct kernel_ kernel,float* Gg);
float* wino23s1d2_BTxB_cpu(float* input_partition,int Cin,float* BTx);
float* elementwise_mm_cpu(float* U,float* V,int Cout,int Cin);
float* wino23s1d2_ATMA_cpu(float* M,int Cout,float* ATM);
float* tile_wino23s1d2_cpu(float* tile_group, struct kernel_ kernel,int Hout,int Wout);
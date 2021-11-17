
// Define winograd transformation matrix.
const float wino23s1d2_BT[4][7] = {{1.0f,0.0f,0.0f,0.0f,-1.0f,0.0f,0.0f},
								   {0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f},
								   {0.0f,0.0f,-1.0f,0.0f,1.0f,0.0f,0.0f},
								   {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,-1.0f} };

const float wino23s1d2_G[4][5] = {{1.0f,0.0f,0.0f,0.0f,0.0f},
								 {1.0f / 2,0.0f,1.0f / 2,0.0f,1.0f / 2},
								 {1.0f / 2,0.0f,-1.0f / 2,0.0f,1.0f / 2},
								 {0.0f,0.0f,0.0f,0.0f,1.0f} };
				 
const float wino23s1d2_AT[3][4] = {{1.0f,1.0f,1.0f,0.0f},
								   {0.0f,0.0f,0.0f,0.0f},
								   {0.0f,1.0f,-1.0f,-1.0f}};


const float wino23s1d2_B[7][4] = {{1.0f,0.0f,0.0f,0.0f},
								  {0.0f,0.0f,0.0f,0.0f},
								  {0.0f,1.0f,-1.0f,1.0f},
								  {0.0f,0.0f,0.0f,0.0f},
								  {-1.0f,1.0f,1.0f,0.0f},
								  {0.0f,0.0f,0.0f,0.0f},
								  {0.0f,0.0f,0.0f,-1.0f}};

const float wino23s1d2_GT[5][4] = {{1.0f,1.0f/2,1.0f/2,0.0f},
								  {0.0f,0.0f,0.0f,0.0f},
								  {0.0f,1.0f/2,-1.0f/2,0.0f},
								  {0.0f,0.0f,0.0f,0.0f},
								  {0.0f,1.0f/2,1.0f/2,1.0f}};
				 
const float wino23s1d2_A[4][3] = {{1.0f,0.0f,0.0f},
								   {1.0f,0.0f,1.0f},
								   {1.0f,0.0f,-1.0f},
								   {0.0f,0.0f,-1.0f}};


// Winograd F(, 5x5), 8x8 input tiles
float* wino23s1d2_GgGT_cpu(struct kernel_ kernel,float* Gg);
float* wino23s1d2_BTxB_cpu(float* input_partition,int Cin,float* BTx);
float* elementwise_mm_cpu(float* U,float* V,int Cout,int Cin);
float* wino23s1d2_ATMA_cpu(float* M,int Cout,float* ATM);
float* tile_wino23s1d2_cpu(float* tile_group, struct kernel_ kernel,int Hout,int Wout);
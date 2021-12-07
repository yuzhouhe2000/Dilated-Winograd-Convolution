#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

struct tensor_
{
    // row major order
    float *data;
    int N,H,W,C;
    unsigned long int SIZE;
};

struct kernel_
{
    // column major order
    float *data;
    int Cout,Cin,H, W, dilH, dilW, padH, padW, strideH, strideW;
    unsigned long int SIZE;
} ;

float* cudaNew_(int size);
void free_(float* ptr);

void print_kernel(struct kernel_ kernel);
void print_tensor(struct tensor_ output);
struct kernel_ kernel_simple_dilation(struct kernel_ kernel);
float* slice(float* input, int start, int end);
int find_kernel_idx(int cout, int cin, int hk, int wk, struct kernel_ kernel);
int find_tensor_idx(int n, int cin, int hin, int win, struct tensor_ input);
int find_NCHW_idx(int n, int cin, int hin, int win, int N, int C, int H, int W);
int find_CCHW_idx(int cout, int cin, int hk, int wk, int Cout, int Cin, int H, int W);
void print_CHW(float* input, int C, int H, int W);
float* transpose(float *input, const int N,const int C,const int H, const int W);
struct tensor_ tensor_pad(struct tensor_ input, int padH, int padW);
int check_tensor(struct tensor_ A, struct tensor_ B);
float* NCHW_2_NHWC(float* input, const int N, const int C, const int H, const int W);
float* NHWC_2_NCHW(float* input, const int N, const int C, const int H, const int W);

// Safe free
void cudaFree_(float* ptr);
// cuda data operation
struct tensor_ tensor2gpu(struct tensor_ input);
struct tensor_ tensor2cpu(struct tensor_ input_gpu);
struct kernel_ kernel2gpu(struct kernel_ kernel);
struct kernel_ kernel2cpu(struct kernel_ kernel_gpu);
float* data2gpu(float* input,unsigned long int SIZE);
float* data2cpu(float* input_gpu,unsigned long int SIZE);

__device__ int find_NCHW_idx_gpu(int n,int cin,int hin,int win,int N,int C,int H,int W);
__device__ int find_tensor_idx_gpu(int n,int cin, int hin,int win,struct tensor_ input);
__device__ int find_kernel_idx_gpu(int cout,int cin,int hk, int wk,struct kernel_ kernel);

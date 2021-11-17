struct tensor_
{
    // row major order
    float *data;
    int N,H,W,C;
    int SIZE;
};

struct kernel_
{
    // column major order
    float *data;
    int Cout,Cin,H, W, dilH, dilW, padH, padW, strideH, strideW;
    int SIZE;
} ;

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


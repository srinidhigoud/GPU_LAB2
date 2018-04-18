#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <cudnn.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 5
#define BLOCK_SIZE 4
#define TILE_SIZE 6
#define RADIUS (TILE_SIZE - BLOCK_SIZE)

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

using namespace std;


void part3(){
	struct timeval t1, t2; 
	double total_t;
	cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
	cudnnTensorDescriptor_t input_descriptor, output_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor;
	const double alpha = 1, beta = 0;
	cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		                              /*format=*/CUDNN_TENSOR_NCHW,
		                              /*dataType=*/CUDNN_DATA_DOUBLE,
		                              /*batch_size=*/1,
		                              /*channels=*/C,
		                              /*image_height=*/H,
		                              /*image_width=*/W));
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
	                                  /*format=*/CUDNN_TENSOR_NCHW,
	                                  /*dataType=*/CUDNN_DATA_DOUBLE,
	                                  /*batch_size=*/1,
	                                  /*channels=*/K,
	                                  /*image_height=*/H,
	                                  /*image_width=*/W));

	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_DOUBLE,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/K,
                                      /*in_channels=*/C,
                                      /*kernel_height=*/FH,
                                      /*kernel_width=*/FW));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                       /*pad_height=*/1,
                                       /*pad_width=*/1,
                                       /*vertical_stride=*/1,
                                       /*horizontal_stride=*/1,
                                       /*dilation_height=*/1,
                                       /*dilation_width=*/1,
                                       /*mode=*/CUDNN_CONVOLUTION,
                                       /*computeType=*/CUDNN_DATA_DOUBLE));
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
	                                               input_descriptor,
	                                               kernel_descriptor,
	                                               convolution_descriptor,
	                                               output_descriptor,
	                                               convolution_algorithm,
	                                               &workspace_bytes));
	// cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << endl;

	void* d_workspace{nullptr};
	cudaMalloc(&d_workspace, workspace_bytes);

	int image_bytes = 1 * C * H * W * sizeof(double);
	int output_bytes = 1 * K * H * W * sizeof(double);
	int kernel_bytes = 1 * K * C * FH * FW * sizeof(double);
	double *image = (double*)malloc(image_bytes);
	double *output = (double*)malloc(output_bytes);
	double *h_kernel = (double*)malloc(kernel_bytes);
	for(int i=0;i<C;i++) for(int j=0;j<H;j++) for(int k=0;k<W;k++) image[i*H*W+j*W+k] = i*(j+k);
	for(int a=0;a<K;a++) for(int b=0;b<C;b++) for(int c=0;c<FH;c++) for(int d=0;d<FW;d++) h_kernel[a*C*FH*FW+b*FH*FW+c*FW+d] = (a+b)*(c+d);
	double* d_input{nullptr};
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, image, image_bytes, cudaMemcpyHostToDevice);

	double* d_output{nullptr};
	cudaMalloc(&d_output, output_bytes);
	cudaMemset(d_output, 0, output_bytes);

	double* d_kernel{nullptr};
	cudaMalloc(&d_kernel, kernel_bytes);
	cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
	gettimeofday(&t1, NULL);

	checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   d_input,
                                   kernel_descriptor,
                                   d_kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   d_output));
	gettimeofday(&t2, NULL);

	cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
	double sum = 0;
	for(int i=0;i<K;i++) {
		for(int j=0;j<H;j++) {
			for(int k=0;k<W;k++){
				sum += output[i*H*W+j*W+k];
				// cout<<output[i*H*W+j*W+k]<<'\t';
			} 
			// cout<<endl;
		} 
		// cout<<endl;
	} 

	total_t = ((double)(t2.tv_sec-t1.tv_sec)*1000 + t2.tv_usec-t1.tv_usec)/1000;


	printf("%4.3lf, %4.3lf ms\n",sum,total_t);

	free(image);
	free(output);
	free(h_kernel);
	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);

	cudnnDestroy(cudnn);
}


void __global__ conv1(double * input, double * output, double * filter){
	int const lrow = blockIdx.y * blockDim.y + threadIdx.y;
	int const lcol = blockIdx.x * blockDim.x + threadIdx.x;
	int const llayer = blockIdx.z * blockDim.z + threadIdx.z;
	int stridex = blockDim.y * gridDim.y;
	int stridey = blockDim.x * gridDim.x;
	int stridez = blockDim.z * gridDim.z;
	
	for(int layer = llayer;layer<K;layer += stridez){
		for(int col = lcol;col<W;col += stridey){
			for(int row = lrow;row<H;row += stridex){
				double Mval = 0;
				for(int i=0;i<C;i++) for(int j=0;j<FH;j++) for(int k=0;k<FW;k++) Mval += filter[layer*C*FH*FW+i*FH*FW+(FH-1-j)*FW+(FW-1-k)]*input[i*(H+2)*(W+2)+(row+k)*(W+2)+(col+j)];
				output[layer*H*W+row*W+col] = Mval;
			} 
		}
	}	
}

void __global__ conv2(double * input, double * output, double * filter){
	int const lrow = blockIdx.y * blockDim.y + threadIdx.y;
	int const lcol = blockIdx.x * blockDim.x + threadIdx.x;
	int row = lrow;
	int col = lcol;
	int const llayer = blockIdx.z * blockDim.z + threadIdx.z;
	int stridex = blockDim.y * gridDim.y;
	int stridey = blockDim.x * gridDim.x;
	int stridez = blockDim.z * gridDim.z;
	int x = threadIdx.y;
	int y = threadIdx.x;
	__shared__ double As[C][TILE_SIZE][TILE_SIZE];
	
	if(row<H+2 && col<W+2){
		if(x<RADIUS || y<RADIUS){
			int temprow = row + blockDim.y;
			int tempx = x + blockDim.y;
			int tempcol = col + blockDim.x;
			int tempy = y + blockDim.x;
			if(x<RADIUS && y<RADIUS && temprow<H+2 && tempcol<W+2){
				for(int i=0;i<C;i++){
					As[i][tempx][y] = input[i*(H+2)*(W+2)+temprow*(W+2)+col];
					As[i][x][tempy] = input[i*(H+2)*(W+2)+row*(W+2)+tempcol];
					As[i][x][y] = input[i*(H+2)*(W+2)+row*(W+2)+col];
					As[i][tempx][tempy] = input[i*(H+2)*(W+2)+temprow*(W+2)+tempcol];
				}
			}
			else if(x<RADIUS && temprow<H+2){
				for(int i=0;i<C;i++){
					As[i][tempx][y] = input[i*(H+2)*(W+2)+temprow*(W+2)+col];
					As[i][x][y] = input[i*(H+2)*(W+2)+row*(W+2)+col];
				}
			}
			else if(y<RADIUS && tempcol<W+2){
				for(int i=0;i<C;i++){
					As[i][x][tempy] = input[i*(H+2)*(W+2)+row*(W+2)+tempcol];
					As[i][x][y] = input[i*(H+2)*(W+2)+row*(W+2)+col];
				}
			}
		}
		else for(int i=0;i<C;i++) As[i][x][y] = input[i*(H+2)*(W+2)+row*(W+2)+col];
	}
	__syncthreads();

	
	for(int layer = llayer;layer<K;layer += stridez){
		for(int col = lcol;col<W;col += stridey){
			for(int row = lrow;row<H;row += stridex){
				double Mval = 0;
				for(int i=0;i<C;i++) for(int j=0;j<FH;j++) for(int k=0;k<FW;k++) Mval += filter[layer*C*FH*FW+i*FH*FW+(FH-1-j)*FW+(FW-1-k)]*As[i][x+k][y+j];
				__syncthreads();
				output[layer*H*W+row*W+col] = Mval;	
			} 
		}
	}	
	
	// __syncthreads();

	// for(int layer = llayer;layer<K;layer += stridez){
	// 	for(int col = lcol;col<W;col += stridey){
	// 		for(int row = lrow;row<H;row += stridex){
	// 			output[layer*H*W+row*W+col] = Mval;
	// 		} 
	// 	}
	// }
}


void part1(){
	int blockDimX = BLOCK_SIZE;              // X ~ COL
    int blockDimY = BLOCK_SIZE;              // Y ~ ROW
    int blockDimZ = BLOCK_SIZE;  			// Z ~ Depth
    struct timeval t1, t2; 
    double total_t;
	int image_bytes = 1 * C * (H+2) * (W+2) * sizeof(double);
	int output_bytes = 1 * K * H * W * sizeof(double);
	int kernel_bytes = 1 * K * C * FH * FW * sizeof(double);
	double *image = (double*)malloc(image_bytes);
	double *output = (double*)malloc(output_bytes);
	double *h_kernel = (double*)malloc(kernel_bytes);
	for(int i=0;i<C;i++) for(int j=0;j<H+2;j++) for(int k=0;k<W+2;k++) image[i*(H+2)*(W+2)+(j)*(W+2)+k] = (j==0 || j==H+1 || k==0 || k==W+1)?0:i*(j-1+k-1);
	for(int a=0;a<K;a++) for(int b=0;b<C;b++) for(int c=0;c<FH;c++) for(int d=0;d<FW;d++) h_kernel[a*C*FH*FW+b*FH*FW+c*FW+d] = (a+b)*(c+d);
	
	double* d_input{nullptr};
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, image, image_bytes, cudaMemcpyHostToDevice);

	double* d_output{nullptr};
	cudaMalloc(&d_output, output_bytes);
	cudaMemset(d_output, 0, output_bytes);

	double* d_kernel{nullptr};
	cudaMalloc(&d_kernel, kernel_bytes);
	cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
	
    dim3 dimBlock(blockDimX, blockDimY, blockDimZ);
    dim3 dimGrid(W/ dimBlock.x, H/ dimBlock.y);
	
	

	gettimeofday(&t1, NULL);
	conv1<<<dimGrid, dimBlock>>>(d_input,d_output,d_kernel);
	
	
	//cudaDeviceSynchronize();
	cudaError_t cudaerr = cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
	double sum = 0;
	for(int i=0;i<K;i++) {
		for(int j=0;j<H;j++) {
			for(int k=0;k<W;k++){
				sum += output[i*H*W+j*W+k];
				// cout<<output[i*H*W+j*W+k]<<'\t';
			} 
			// cout<<endl;
		} 
		// cout<<endl;
	} 
	total_t = ((double)(t2.tv_sec-t1.tv_sec)*1000 + t2.tv_usec-t1.tv_usec)/1000;


	printf("%4.3lf, %4.3lf ms\n",sum,total_t);
	
	
	free(image);
	free(output);
	free(h_kernel);
	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
}

void part2(){
	int blockDimX = BLOCK_SIZE;              // X ~ COL
    int blockDimY = BLOCK_SIZE;              // Y ~ ROW
    int blockDimZ = BLOCK_SIZE;  			// Z ~ Depth
    struct timeval t1, t2; 
    double total_t;
	int image_bytes = 1 * C * (H+2) * (W+2) * sizeof(double);
	int output_bytes = 1 * K * H * W * sizeof(double);
	int kernel_bytes = 1 * K * C * FH * FW * sizeof(double);
	double *image = (double*)malloc(image_bytes);
	double *output = (double*)malloc(output_bytes);
	double *h_kernel = (double*)malloc(kernel_bytes);
	for(int i=0;i<C;i++) for(int j=0;j<H+2;j++) for(int k=0;k<W+2;k++) image[i*(H+2)*(W+2)+(j)*(W+2)+k] = (j==0 || j==H+1 || k==0 || k==W+1)?0:i*(j-1+k-1);
	for(int a=0;a<K;a++) for(int b=0;b<C;b++) for(int c=0;c<FH;c++) for(int d=0;d<FW;d++) h_kernel[a*C*FH*FW+b*FH*FW+c*FW+d] = (a+b)*(c+d);
	
	double* d_input;
	cudaMalloc((void**)&d_input, image_bytes);
	cudaMemcpy(d_input, image, image_bytes, cudaMemcpyHostToDevice);

	double* d_output;
	cudaMalloc(&d_output, output_bytes);
	cudaMemset(d_output, 0, output_bytes);

	double* d_kernel;
	cudaMalloc(&d_kernel, kernel_bytes);
	cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
	
    dim3 dimBlock(blockDimX, blockDimY, blockDimZ);
    dim3 dimGrid(W/ dimBlock.x, H/ dimBlock.y);
	
	

	gettimeofday(&t1, NULL);
	conv2<<<dimGrid, dimBlock>>>(d_input,d_output,d_kernel);
	
	//cudaDeviceSynchronize();
	cudaError_t cudaerr = cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
	double sum = 0;
	for(int i=0;i<K;i++) {
		for(int j=0;j<H;j++) {
			for(int k=0;k<W;k++){
				sum += output[i*H*W+j*W+k];
				// cout<<output[i*H*W+j*W+k]<<'\t';
			} 
			// cout<<endl;
		} 
		// cout<<endl;
	} 
	total_t = ((double)(t2.tv_sec-t1.tv_sec)*1000 + t2.tv_usec-t1.tv_usec)/1000;

	printf("%4.3lf, %4.3lf ms\n",sum,total_t);
	
	
	free(image);
	free(output);
	free(h_kernel);
	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
}


int main(){

	part1();
	part2();
	part3();

}

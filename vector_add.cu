
/*
A simple example demonstrating how to write a cuda kernel
The kernel adds 2 arrays and saves the output to a 3rd array

First, we show the output using a serial addition through serial_add_arrays

Then, we show the same output using the function parallel_add_arrays which is a stub function
that invokes vecAddKernel. Inside parallel_add_arrays, first we allocate the required memory
in host (GPU) for the 3 arrays. We can invoke the kernel with a block_size of 32. In other words,
we request 32 thread for each block. Finally, we copy the result in 3rd array back from
device (GPU) to host (CPU) memory and free up the host memory.

Commands to run on a Linux machine and the output:
nvcc -c vector_ad.cu
nvcc vector_add.o -o vector_add
./vector_add

Output of serial add: 4, 6, 4, 11, 14, 
Output of parallel add: 4, 6, 4, 11, 14,
*/

#include<cuda.h>
#include<stdio.h>

__global__
void vecAddKernel(float *a, float *b, float *c, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

void parallel_add_arrays(float *a, float *b, float *c, int n){
    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_b, size);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_c, size);

    int block_size = 32;
    vecAddKernel<<<ceil(n/(float)block_size), block_size>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void serial_add_arrays(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void print_arr(float *a, int n, char *header){
    std::cout<<header;
    for(int i = 0; i < n; i++)
        std::cout<<a[i]<<", ";
    std::cout<<std::endl;
}

int main(){
    float A[5] = {1, 2, 3, 4, 5};
    float B[5] = {3, 4, 1, 7, 9};
    float C[5];
    int array_size = 5;

    serial_add_arrays(A, B, C, array_size);
    print_arr(C, array_size, "Output of serial add: ");
    parallel_add_arrays(A, B, C, array_size);
    print_arr(C, array_size, "Output of parallel add: ");

    return 0;
};
#include <iostream>
#include <cuda.h>
#include <cstdio>
#include<unistd.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda::wmma;

const int WMMA_M = 16;
const int WMMA_K = 16;
const int WMMA_N = 16;

void __global__ init(half * a, half * b ){
    int tid = blockIdx.x * 32 * 32 + threadIdx.x;

    a[tid] = tid;
    b[tid] = tid;
}

void __global__ matrixmul(half * A, half * B, float * C, int M, int K, int N){
    
        
    fragment<matrix_a, 16, 16, 16, half, row_major> aFrag;
    fragment<matrix_b, 16, 16, 16, half, row_major> bFrag;
    fragment<accumulator, 16, 16, 16, float> cFrag;

    // 16 for loop
    // (m * n)  = (m * k) * (k * n) + (m * n) 
    for(int i = 0; i < M/WMMA_M; i++){
        for(int j = 0 ; j < N/WMMA_M; j++){
            // i * 64 * 16 + j * 16
            int C_index = i * N * WMMA_M + j * WMMA_N;
            // i * 64 * 16 + j * 16
            fill_fragment(cFrag, 0.0f);

            for(int k = 0; k < K/WMMA_K; k++){
                int aind = (i * K * WMMA_M) + k * WMMA_K;
                int bind = j * WMMA_N + k * N * WMMA_M ;
                load_matrix_sync(bFrag, B + bind, N);
                
                load_matrix_sync(aFrag, A + aind, K);

                mma_sync(cFrag, aFrag, bFrag, cFrag);

            } 
            store_matrix_sync(C + C_index, cFrag, N, mem_row_major);
        }
    }


}

int main(){
    half *a;
    half *b;
    float *c;

    half *dev_a;
    half *dev_b;
    float *dev_c;

    // a = m * k , b = k * n , c = m * n 
    int m = 64, k = 64, n = 64;

    cudaMallocHost(&a, m * k * sizeof(half));
    cudaMallocHost(&b, k * n * sizeof(half));
    cudaMallocHost(&c, m * n * sizeof(float));

    cudaMalloc(&dev_a, m * k * sizeof(half));
    cudaMalloc(&dev_b, k * n * sizeof(half));
    cudaMalloc(&dev_c, m * n * sizeof(float));

    init<<<4, 1024>>>(dev_a, dev_b);

    
    cudaError_t err;
    
    err = cudaGetLastError();
    
    if(err != cudaSuccess){
        printf("Error while running init %s", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    
    matrixmul<<<1, 32>>>(dev_a, dev_b, dev_c, m, k, n);

    err = cudaGetLastError();
    
    if(err != cudaSuccess){
        printf("Error while running matrixmul %s", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaDeviceSynchronize();

    if(err != cudaSuccess){
        printf("Error while running Synchronising %s", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(c, dev_c, m * n * sizeof(float), cudaMemcpyDefault);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            std::cout << c[i*n + j] << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}
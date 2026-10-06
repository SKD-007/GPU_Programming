#include <stdio.h>
#include <cuda_runtime.h>


__device__ int val = 0;


__device__ int blocks_executed = 0;

__global__ void func(int TotalNoBlocks){

    if(threadIdx.x == 0){
        atomicAdd(&blocks_executed, 1);
    }
    
    printf("Block Id = %d, ThreadID = %d\n", blockIdx.x, threadIdx.x);

    while(blocks_executed < TotalNoBlocks ){
        continue;
    }

    __syncthreads();

    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("This will be printed at last\n");
    }

}

int main(){

    int NoBlock = 20;
    int NoThreadperBlock = 128;
    
	func <<<NoBlock, NoThreadperBlock>>> (NoBlock);
    cudaDeviceSynchronize();

    return 0;
}

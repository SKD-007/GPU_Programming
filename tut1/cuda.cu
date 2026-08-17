#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {
    // Defines cores per SM based on architecture generation
    switch (major) {
		case 2: // Fermi
            return (minor == 1) ? 48 : 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            if (minor == 1 || minor == 2) return 128;
            if (minor == 0) return 64;
            return 128; // Default fallback for Pascal
        case 7: // Volta (7.0), Turing (7.5)
            return 64;
        case 8: // Ampere (8.0, 8.6, 8.7), Ada Lovelace (8.9)
            if (minor == 0) return 64;
            if (minor == 6 || minor == 9) return 128;
            return 64; // Default fallback for Ampere variants
        case 9: // Hopper (9.0), Blackwell (9.5)
            return 128;
        default:
            return 128; // Standard fallback for future architectures
    }
}

int main(){

	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	if( error != cudaSuccess){
		printf("CUDA error : %s\n", cudaGetErrorString(error));
		return 1;
	}

	printf("Number of Device Found : %d\n", deviceCount);

	for(int deviceid = 0; deviceid < deviceCount; deviceid++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceid);

		printf("Device Name : %s\n", prop.name);
		printf("Computing Capability (major.minor) : %d.%d\n", prop.major, prop.minor);
		printf("Global Memory : %lf GB\n", (double)prop.totalGlobalMem/(1024*1024*1024) );
		printf("Streaming Mulitprocessors (SMs) Count : %d\n", prop.multiProcessorCount);
		printf("Cores Per SMs : %d\n", getCoresPerSM(prop.major, prop.minor));
		printf("Total Number of Cores : %d\n", prop.multiProcessorCount*getCoresPerSM(prop.major, prop.minor));
		printf("Max Thread Per Block : %d\n", prop.maxThreadsPerBlock);
		printf("Shared Memory Per Block : %lu KB\n", prop.sharedMemPerBlock/1024 );
		printf("Warp Size : %d\n", prop.warpSize);

		printf("L2 Cache size : %lf MB\n", (double)prop.l2CacheSize/(1024*1024));
		printf("Max Grid Size along each dimension X : %d, Y: %d, Z : %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Max Block size  along each dimension X : %d, Y: %d, Z : %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Thread along each Block : %d\n", prop.maxThreadsPerBlock);
		printf("Maximum resident threads per multiprocessor : %d\n", prop.maxThreadsPerMultiProcessor);
		printf("Maximum pitch in bytes allowed by memory copies : %lf\n", (double)prop.memPitch/(1024*1024));
		printf("Maximum persistent memory capacity availble for L2 Cache : %lf KB\n", (double)prop.persistingL2CacheMaxSize/(1024));
		printf("Registers available per BLock : %d\n", prop.regsPerBlock);
		printf("Registers available per SM : %d\n", prop.regsPerMultiprocessor);
		printf("Shared memory available per block in bytes : %d", prop.sharedMemPerBlock);		
	}	

	return 0;
}

#include <stdio.h>

__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Print from host (CPU)
    printf("Hello World from CPU!\n");

    // Launch kernel with 1 block and 1 thread
    cuda_hello<<<1, 1>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}
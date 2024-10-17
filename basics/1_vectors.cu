#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define RANGE 17.78

// CUDA kernel function declaration
__global__ void vecGPU(float *ad, float *bd, float *cd, int n);

int main(int argc, char *argv[]) {
    int n = 0;  // Number of elements in the arrays
    int i;      // Loop index
    float *a, *b, *c;  // Host arrays
    float *temp;       // Temporary array for CPU computation
    float *ad, *bd, *cd;  // Device arrays
    clock_t start, end;   // For timing measurements

    if(argc != 2) {
        printf("usage:  ./vectorprog n\n");
        printf("n = number of elements in each vector\n");
        exit(1);
    }
        
    n = atoi(argv[1]);
    int num_blocks = 8;
    int threads_per_block = 500;

    printf("Each vector will have %d elements\n", n);
    
    // Allocate memory for host arrays
    if (!(a = (float *)malloc(n*sizeof(float))) ||
        !(b = (float *)malloc(n*sizeof(float))) ||
        !(c = (float *)malloc(n*sizeof(float))) ||
        !(temp = (float *)malloc(n*sizeof(float)))) {
        printf("Error allocating host memory\n");
        exit(1);
    }
    
    // Initialize arrays with random values
    srand((unsigned int)time(NULL));
    for (i = 0; i < n; i++) {
        a[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        b[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        c[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        temp[i] = c[i];  // temp is a copy of c for CPU computation
    }
    
    // Perform CPU (sequential) computation
    start = clock();
    for(i = 0; i < n; i++)
        temp[i] += a[i] * b[i];
    end = clock();
    printf("Total time taken by the sequential part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Allocate memory on the GPU
    cudaMalloc((void**)&ad, n * sizeof(float));
    cudaMalloc((void**)&bd, n * sizeof(float));
    cudaMalloc((void**)&cd, n * sizeof(float));

    // Start timing for GPU computation
    start = clock();

    // Copy data from host to device
    cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd, c, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the CUDA kernel
    vecGPU<<<num_blocks, threads_per_block>>>(ad, bd, cd, n);

    // Copy result back to host
    cudaMemcpy(c, cd, n * sizeof(float), cudaMemcpyDeviceToHost);

    end = clock();

    // Free GPU memory
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    
    printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    // Verify results
    for(i = 0; i < n; i++)
      if (fabs(temp[i] - c[i]) >= 0.009)
        printf("Element %d in the result array does not match the sequential version\n", i);
        
    // Free host memory
    free(a); free(b); free(c); free(temp);

    return 0;
}

// CUDA kernel function
__global__ void vecGPU(float *ad, float *bd, float *cd, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread ID
    int stride = blockDim.x * gridDim.x;              // Calculate stride for grid-stride loop

    // Grid-stride loop
    for (int i = tid; i < n; i += stride) {
        cd[i] += ad[i] * bd[i];  // Vector multiplication
    }
}
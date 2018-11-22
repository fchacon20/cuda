#include <iostream>
#include <tgmath.h>

using namespace std;

__device__ volatile int count = 0;


__global__ kernel(int N, devA, devB, devC){
    if (tId < N){
        if((count % 500) == 0){
            if (tId == 0)
                devC[tId] = devB[tId + N - 1] + devB[tId] + devB[tId + 1];
            else if (tId == N - 1)
                devC[tId] = devB[tId - 1] + devB[tId] + devB[tId - N + 1];
            else
                devC[tId] = devB[tId - 1] + devB[tId] + devB[tId + 1];
        } else {
            if ((count % 2) == 0) {
                if (tId == 0)
                    devB[tId] = devA[tId + N - 1] + devA[tId] + devA[tId + 1];
                else if (tId == N - 1)
                    devB[tId] = devA[tId - 1] + devA[tId] + devA[tId - N + 1];
                else
                    devB[tId] = devA[tId - 1] + devA[tId] + devA[tId + 1];
            } else {
                if (tId == 0)
                    devA[tId] = devB[tId + N - 1] + devB[tId] + devB[tId + 1];
                else if (tId == N - 1)
                    devA[tId] = devB[tId - 1] + devB[tId] + devB[tId - N + 1];
                else
                    devA[tId] = devB[tId - 1] + devB[tId] + devB[tId + 1];
            }
        }
        count++;
    }
}

void fillArray(int *a, int n){
    for (int i = 0; i < n; ++i)
        a[i] = 1;
}

int main() {

    int N = (int) 1e8;
    int *a;
    int *devA, *devB, *devC;

    cudaMallocHost(&a, N *sizeof(int));
    cudaMalloc(&devA, N * sizeof(int));
    cudaMalloc(&devB, N * sizeof(int));
    cudaMalloc(&devC, N * sizeof(int));

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    fillArray(a, N);

    cudaMemcpy(devA, a, N*sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (int)ceil((float) N/block_size);

    for (int i = 0; i < 4e4; ++i) {
        kernel<<<grid_size, block_size, 0, stream1>>>(N, devA, devB, devC);
        if (i % 500 == 0)
            cudaDeviceSynchronize();
            cudaMemcpyAsync(a, devA, N*sizeof(int), cudaMemcpyDeviceToHost, stream2);
    }

    cudaMemcpy(a, devA, N*sizeof(int), cudaMemcpyDeviceToHost);

    /*for (int i = 0; i < 10; ++i) {
        cout << a[i] << endl;
    }*/

    delete a;
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
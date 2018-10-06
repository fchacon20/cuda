
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>

#define E 2.71828182845904523536

using namespace std;

void __global__ calculoGPU(float t0, float y0, float deltaT, int n, float * soluciones_GPU){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    float sum = 0;
    if(tId == 0)
        soluciones_GPU[0] = y0;
    else if(tId < n){
        for(i = 1; i < tId; i++)
            sum = sum + powf(E, -1 * (i - 1) * deltaT);
        soluciones_GPU[tId] = (sum * deltaT) + y0;
    }
}

void eulerMethodGPU(float t0, float y0, float deltaT){
    FILE * archivo;
    float tN = 10;
    int i;
    cudaEvent_t ct1, ct2;
    float dt;
    int n = (tN - t0)/deltaT;
    int block_size = 256;
    int grid_size = (int)ceil((float)n / block_size);
    float * soluciones = new float[n];
    float * soluciones_GPU;
    cudaMalloc(&soluciones_GPU, n * sizeof(float));
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    calculoGPU << <grid_size, block_size>> >(t0, y0, deltaT, n, soluciones_GPU);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(soluciones, soluciones_GPU, n * sizeof(float), cudaMemcpyDeviceToHost);

	cout << "[GPU] DeltaT = " << deltaT << ": " << dt << " [ms]" << endl;

    archivo = fopen("eulerGPU.txt", "a");
    for(i = 0; i < n; i++){
        if(i == n-1)
            fprintf(archivo, "%f\n", soluciones[i]);
        else
            fprintf(archivo, "%f ", soluciones[i]);
    }
    fclose(archivo);
    cudaFree(soluciones_GPU);
    delete soluciones;
}

float* eulerMethodCPU(int t0, float y0, float deltaT) {
	int n = int(10 / deltaT);
	float* y = new float[n];
	float sum = 0.0;
	
	y[0] = y0;
	for (int i = 1; i < n; i++)	{
		sum = sum + powf(E, -1 *(i-1)*deltaT);
		y[i] = y0 + deltaT * sum;
	}
	return y;
}


int main() {
	
	float* y;
	float deltaT[6] = { 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 };
	clock_t t1;
	clock_t t2;
	long duration;

	ofstream outputFile;
	outputFile.open("euler.txt");

	for (int i = 0; i < 6; i++) {
		t1 = clock();
		y = eulerMethodCPU(0, -1, deltaT[i]);
		t2 = clock();
		duration = 1000*(double)(t2 - t1) / CLOCKS_PER_SEC;
		cout << "[CPU] DeltaT = " << deltaT[i] << ": " << duration << " [ms]" << endl;

		for (int j = 0; j < int(10 / deltaT[i])-1; j++)
			outputFile << y[j] << " ";
		outputFile << y[int(10 / deltaT[i]) - 1] << endl;
	}	
	cout << "---------------" << endl;
	eulerMethodGPU(0, -1, 0.1);
    eulerMethodGPU(0, -1, 0.01);
    eulerMethodGPU(0, -1, 0.001);
	eulerMethodGPU(0, -1, 0.0001);
	
	outputFile.close();
	delete[] y;
	return 0;
}
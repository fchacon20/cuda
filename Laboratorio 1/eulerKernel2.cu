#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>

#define E 2.71828182845904523536

using namespace std;

void actualizadorCPU(float * soluciones, int m, float deltaT, float tI) {
	for (int j = 0; j < m; j++)
		soluciones[j] = soluciones[j] + deltaT * (4 * (tI - deltaT) - soluciones[j] + 3 + j);
}

__global__ void actualizadorGPU(float * devSoluciones, int m, float deltaT, float tI) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < m)
		devSoluciones[tId] = devSoluciones[tId] + deltaT * (4 * (tI - deltaT) - devSoluciones[tId] + 3 + tId);
}

void secondCPU_GPU(int m) {
	int n = 1000;
	double duration;
	float deltaT = (float)0.001;
	clock_t t1, t2;

	float * soluciones = new float[m];
	for (int j = 0; j < m; j++)
		soluciones[j] = (float)j;

	float * devSoluciones;
	cudaMalloc(&devSoluciones, m * sizeof(float));
	cudaMemcpy(devSoluciones, soluciones, n * sizeof(float), cudaMemcpyHostToDevice);

	t1 = clock();
	for (int i = 1; i < n; i++)
		actualizadorCPU(soluciones, m, deltaT, deltaT*i);
	t2 = clock();
	duration = 1000 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	cout << "[CPU] Tamaño m = " << m << ": " << duration << " [ms]" << endl;

	cudaEvent_t ct1, ct2;
	float dt;
	int grid_size = (int)ceil((float)m / 256);
	int block_size(256);
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);
	for (int i = 1; i < n; i++)
		actualizadorGPU << <grid_size, block_size >> > (devSoluciones, m, deltaT, deltaT*i);
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	cudaMemcpy(soluciones, devSoluciones, m * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "[GPU] Tamaño m = " << m << ": " << dt << " [ms]" << endl;

	cudaFree(devSoluciones);
	delete[] soluciones;
}

void fixedM(float* soluciones, float* devSoluciones, int nThreads, int m, int n) {

	float deltaT = (float) 0.001;

	float dt;
	cudaEvent_t ct1, ct2;
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);

	int grid_size = (int)ceil((float)m / nThreads);
	int block_size(nThreads);
	cudaEventRecord(ct1);
	for (int i = 1; i < n; i++)
		actualizadorGPU << <grid_size, block_size >> > (devSoluciones, m, deltaT, deltaT*i);
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	cout << "[GPU] Hebras = " << nThreads << ": " << dt << " [ms]" << endl;

	//cudaMemcpy(soluciones, devSoluciones, m * sizeof(float), cudaMemcpyDeviceToHost);

}

int main() {

	secondCPU_GPU((int)1e4);
	secondCPU_GPU((int)1e5);
	secondCPU_GPU((int)1e6);
	secondCPU_GPU((int)1e7);
	secondCPU_GPU((int)1e8);

	int n = 1000;
	int m = (int) 1e8;

	float * soluciones = new float[m];
	for (int j = 0; j < m; j++)
		soluciones[j] = (float)j;

	float * devSoluciones;
	cudaMalloc(&devSoluciones, m * sizeof(float));
	cudaMemcpy(devSoluciones, soluciones, m * sizeof(float), cudaMemcpyHostToDevice);

	cout << "-----------" << endl;

	fixedM(soluciones, devSoluciones, 64,  m, n);
	fixedM(soluciones, devSoluciones, 128, m, n);
	fixedM(soluciones, devSoluciones, 256, m, n);
	fixedM(soluciones, devSoluciones, 512, m, n);

	cudaFree(devSoluciones);
	delete[] soluciones;
	return 0;
}
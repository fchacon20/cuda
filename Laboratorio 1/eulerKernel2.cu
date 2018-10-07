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
	if (tId < m) {
		devSoluciones[tId] = devSoluciones[tId] + deltaT * (4 * (tI - deltaT) - devSoluciones[tId] + 3 + tId);
	}
}

void secondCPU_GPU(int m) {
	int n = 1000;
	long duration;
	float deltaT = 0.001;
	clock_t t1, t2;

	float * soluciones = new float[m];
	for (int j = 0; j < m; j++)
		soluciones[j] = j;

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
	cout << "[GPU] Tamaño m = : " << m << ": " << dt << " [ms]" << endl;

	cudaFree(devSoluciones);
	delete soluciones;
}

void fixedM() {
	int n = 1000;
	int m = 1e8;
	float deltaT = 0.001;
	int threads[4] = { 64, 128, 256, 512 };

	float * soluciones = new float[m];
	for (int j = 0; j < m; j++)
		soluciones[j] = j;

	float * devSoluciones;
	cudaMalloc(&devSoluciones, m * sizeof(float));

	cudaEvent_t ct1, ct2;
	float dt;
	for (int t = 0; t < 4; t++) {
		cudaMemcpy(devSoluciones, soluciones, m * sizeof(float), cudaMemcpyHostToDevice);
		int grid_size = (int)ceil((float)m / threads[t]);
		int block_size(threads[t]);
		cudaEventCreate(&ct1);
		cudaEventCreate(&ct2);
		cudaEventRecord(ct1);
		for (int i = 1; i < n; i++)
			actualizadorGPU << <grid_size, block_size >> > (devSoluciones, m, deltaT, deltaT*i);
		cudaEventRecord(ct2);
		cudaEventSynchronize(ct2);
		cudaEventElapsedTime(&dt, ct1, ct2);
		cudaMemcpy(soluciones, devSoluciones, m * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "[GPU] Hebras = : " << threads[t] << ": " << dt << " [ms]" << endl;
	}

	cudaFree(devSoluciones);
	delete soluciones;
}

int main() {

	//secondCPU_GPU(10000);
	//secondCPU_GPU(100000);
	//secondCPU_GPU(1000000);
	//secondCPU_GPU(10000000);
	//secondCPU_GPU(100000000);
	fixedM();

	return 0;
}
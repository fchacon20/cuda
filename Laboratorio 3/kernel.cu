
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

// consideraciones

/*
Se trabajar�a con una matriz cuadrada de 10^8 elementos, i.e. N = M = 10^4

El tamano de un bloque de hebras siempre sera 256.

Los valores especificados en los dos puntos anteriores pueden ser declarados como constantes en
tiempo de compilacion.

La matriz A y el vector x pueden ser inicializados con los valores que guste mientras no contengan
valores nulos (0). Como consejo, se le recomienda que todos los valores sean 1 para entonces poder
comprobar si el resultado es correcto cuando todos los valores en b sean 10^4.
*/

__global__ void kernelA(int *A, int*x, int*b, int N) {
	// Este kernel utiliza N x N = 10^8 hebras. Cada hebra esta asociada a un elemento a_i,j de la matriz A
	// multiplic�ndolo por el valor x_j correspondiente y sumando este resultado al elemento en la i-�sima
	// posici�n del vector b.
	
	int tId = threadIdx.x + blockIdx.x * blockDim.x;

	int i = tId / N;
	int j = tId % N;
	if (i < N && j < N) {
		int mult = A[j+i*N] * x[j];
		atomicAdd(&b[i], mult);
	}
}

__global__ void kernelx(int *A, int*x, int*b, int N) {
	// Este kernel utiliza N = 10^4 hebras. Cada hebra esta asociada a un elemento x_j del vector x,
	// sumando a cada uno de los N valores b_i	, la multiplicacion de dicho x_j
	// por el correspondiente elemento a_i,j

	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < N) {
		int j = tId;
		for (int i = 0; i < N; i++) {
			int mult = x[j] * A[i*N + j];
			atomicAdd(&b[i], mult);
		}
	}
}

__global__ void kernelb(int *A, int*x, int*b, int N) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < N) {
		int i = tId;
		for (int j = 0; j < N; j++) {
			int mult = x[j] * A[j+i*N];
			b[i] += mult;
		}
	}
}

__global__ void kernelRed(int *A, int*x, int*b, int N) {
}

__global__ void kernelSM(int *A, int*x, int*b, int N) {
}

__constant__ int constX[2];
__global__ void kernelCM(int *A, int*b, int N) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < N) {
		int i = tId;
		for (int j = 0; j < N; j++) {
			int mult = constX[j] * A[j + i * N];
			b[i] += mult;
		}
	}
}

void fillArray(int *a, int n) {
	for (int i = 0; i < n; ++i)
		a[i] = 1;
}

int main() {

	int N = 2;
	int M = 2;
	int* A = new int[N*M];
	int* X = new int[M];
	int* B = new int[N];

	int *devA, *devX, *devB;
	cudaMalloc(&devA, N*M * sizeof(int));
	cudaMalloc(&devX, M * sizeof(int));
	cudaMalloc(&devB, N * sizeof(int));

	//fillArray(A, N*M);
	//fillArray(X, N);

	// Test 2x2
	A[0] = 2; A[1] = -1;
	A[2] = 3; A[3] = 5;
	X[0] = 3; X[1] = -4;
	// Resultado es (10, -11)^T

	cudaMemcpy(devA, A, N*M * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devX, X, M * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(devB, 0, N * sizeof(int));

	int block_size = 256;

	// KernelA
	int grid_sizeA = (int)ceil((float)N*M / block_size);
	kernelA << <grid_sizeA, block_size >> > (devA, devX, devB, N);
	cudaMemcpy(B, devB, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(devB, 0, N * sizeof(int));

	cout << "KernelA: " << endl;;
	for (int i = 0; i < N; i++)
		cout << B[i] << endl;
	cout << endl;

	// KernelX
	int grid_sizeX = (int)ceil((float)M / block_size);
	kernelx << <grid_sizeX, block_size >> > (devA, devX, devB, M);
	cudaMemcpy(B, devB, M * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(devB, 0, N * sizeof(int));

	cout << "KernelX: " << endl;;
	for (int i = 0; i < N; i++)
		cout << B[i] << endl;
	cout << endl;

	// KernelB
	int grid_sizeB = (int)ceil((float)N / block_size);
	kernelb << <grid_sizeB, block_size >> > (devA, devX, devB, N);
	cudaMemcpy(B, devB, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(devB, 0, N * sizeof(int));

	cout << "KernelB: " << endl;;
	for (int i = 0; i < N; i++)
		cout << B[i] << endl;
	cout << endl;
	
	// KernelCM
	cudaMemcpyToSymbol(constX, X, M * sizeof(int), 0, cudaMemcpyHostToDevice);
	int grid_sizeCM = (int)ceil((float)N / block_size);
	kernelCM << <grid_sizeCM, block_size >> > (devA, devB, N);
	cudaMemcpy(B, devB, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(devB, 0, N * sizeof(int));

	cout << "KernelCM: " << endl;;
	for (int i = 0; i < N; i++)
		cout << B[i] << endl;
	cout << endl;

	cudaFree(devA);
	cudaFree(devX);
	cudaFree(devB);
	delete A;
	delete X;
	delete B;

	return 0;
}
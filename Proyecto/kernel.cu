#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void lagrange_uno(const float * __restrict__ X, const float * __restrict__ Y, const float * __restrict__ N_x, float* N_y, int N, int M) {

	int tId = threadIdx.x + blockIdx.x * blockDim.x;

	if (tId < N) {

		float sum = 0;
		float prod;

		for (int i = 0; i < M; i++) {
			prod = 1;
			//if (tId == i) continue;

			for (int j = 0; j < M; j++) {
				if (j == i) continue;
				prod = prod * (N_x[tId] - X[i]) / (X[j] - X[i]);
			}

			sum = sum + prod * Y[i];
		}

		// escribir valor de sum a matriz de y

	}

}

int main() {

	int N = 1000000; // Cantidad de puntos a interpolar
	int M = 30; // Cantidad de puntos a utilizar de la función original

	float *X, *Y;		// Arreglos conteniendo coordenadas X e Y de puntos de la función original
	float *N_x, *N_y;	// Arreglos conteniendo coordenadas X e Y de puntos a interpolar.
						// N_x deben generarse, N_y deben calcularse usando el kernel

	return 0;
}
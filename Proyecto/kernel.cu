#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

void initialPoints(float *x, float *y, int n) {

	for (int i = 0; i < n; ++i) {
		x[i] = i;
		y[i] = 3 * x[i] + (float)pow(x[i], 2) + 1;
	}

}

void generateX(float *x, int n) {

	for (int i = 1; i <= n; ++i) {
		x[i] = (float) 0.00003*i;
	}

}

__global__ void lagrange_uno(const float * __restrict__ X, const float * __restrict__ Y, const float * __restrict__ N_x, float* N_y, int N, int M) {

	int tId = threadIdx.x + blockIdx.x * blockDim.x;

	if (tId < N) {

		float sum = 0;
		float prod;

		for (int i = 0; i < M; i++) {
			prod = 1;
			if (N_x[tId] == X[i]) continue;

			for (int j = 0; j < M; j++) {
				if (j == i) continue;
				prod = prod * (N_x[tId] - X[i]) / (X[j] - X[i]);
			}

			sum = sum + prod * Y[i];
		}

		// escribir valor de y al arreglo
		N_y[tId] = sum;

	}

}

int main() {
	int N = 1000000; // Cantidad de puntos a interpolar
	int M = 30; // Cantidad de puntos a utilizar de la función original

	float *X, *Y;		// Arreglos conteniendo coordenadas X e Y de puntos de la función original
	float *N_x, *N_y;	// Arreglos conteniendo coordenadas X e Y de puntos a interpolar.
						// N_x deben generarse, N_y deben calcularse usando el kernel

	float *x = new float[M]; // coordenadas x de la función
	float *y = new float[M]; // coordenadas y de la función
	float *x_generados = new float[N];
	float *y_generados = new float[N];
	
	initialPoints(x, y, M);
	generateX(x_generados, N);

	// Saving input
	//ofstream outfile("C:/Usuarios/Wil/Escritorio/Wil/initialPoints.txt");
	ofstream outfile("./initialPoints.txt");
	for (int i = 0; i < M - 1; ++i)
		outfile << x[i] << ",";
	outfile << x[M - 1] << "\n";
	for (int i = 0; i < M - 1; ++i)
		outfile << y[i] << ",";
	outfile << y[M - 1] << "\n";
	outfile.close();

	int block_size = 256;
	int grid_size = (int)ceil((float)N / block_size);

	cudaEvent_t ct1, ct2;
	float dt;

	cudaMalloc(&X, M * sizeof(float));
	cudaMalloc(&Y, M * sizeof(float));
	cudaMalloc(&N_x, N * sizeof(float));
	cudaMalloc(&N_y, N * sizeof(float));

	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);

	cudaMemcpy(X, x, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Y, y, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(N_x, x_generados, N * sizeof(float), cudaMemcpyHostToDevice);

	lagrange_uno << < grid_size, block_size >> > (X, Y, N_x, N_y, N, M);
	cudaMemcpy(y_generados, N_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);

	cout << "[GPU] Duration: " << dt << " ms" << endl;

	ofstream outfile2("./output.txt");
	for (int i = 0; i < N - 1; ++i)
		outfile2 << x_generados[i] << ",";
	outfile2 << x_generados[N - 1] << "\n";
	for (int i = 0; i < N - 1; ++i)
		outfile2 << y_generados[i] << ",";
	outfile2 << y_generados[N - 1] << "\n";
	outfile2.close();

	cudaFree(X);
	cudaFree(Y);
	cudaFree(N_x);
	cudaFree(N_y);
	
	delete x;
	delete y;
	delete x_generados;
	delete y_generados;

	return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

void initialPoints(float *x, float *y, int M) {
	for (int i = 0; i < M; ++i) {
        x[i] = i;
		y[i] = 3 * x[i] + (float)pow(x[i], 2) + 1;
	}
}

void generateX(float *x_generados, int N) {
	for (int i = 1; i <= N; ++i)
		x_generados[i-1] = (float) 0.00003*i;
}

void calculateWeights(double * weights, float * x, int M){
    for (int i = 0; i < M; i++){
        weights[i] = 1;
        for (int j = 0; j < M; j++){
            if(i != j)
                weights[i] *= (x[i] - x[j]);
        }
        weights[i] = (double)pow(weights[i], -1);
    }
}

__global__ void BaryKernel(const float * __restrict__ X, const float * __restrict__ Y, const float * __restrict__ N_x, 
                           const double* __restrict__ devWeights, float* N_y, int N, int M){

    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    if(tId < N){
        double num = 0;
        double den = 0;
        
        for(int j = 0; j < N; j++){
            num += (devWeights[j]/(N_x[tId] - X[j])) * Y[j];
            den += (devWeights[j]/(N_x[tId] - X[j]));
        }
        N_y[tId] = num / den;
    }
}

int main(){

    int N = 100000; // Cantidad de puntos a interpolar
    int M = 30; // Cantidad de puntos a utilizar de la funci�n original
    double * weights = new double[M];
    cudaEvent_t ct1, ct2;
    float dt;

	float *X, *Y;		// Arreglos conteniendo coordenadas X e Y de puntos de la funci�n original
	float *N_x, *N_y;	// Arreglos conteniendo coordenadas X e Y de puntos a interpolar.
						// N_x deben generarse, N_y deben calcularse usando el kernel
    double *devWeights;

	float *x = new float[M]; // coordenadas x de la funci�n
	float *y = new float[M]; // coordenadas y de la funci�n
	float *x_generados = new float[N];
    float *y_generados = new float[N];

    initialPoints(x, y, M);
    generateX(x_generados, N);
    calculateWeights(weights, x, M);
    
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
    
    cudaMalloc(&X, M * sizeof(float));
	cudaMemcpy(X, x, M* sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&Y, M * sizeof(float));
	cudaMemcpy(Y, y, M * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&devWeights, M * sizeof(double));
    cudaMemcpy(devWeights, weights, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&N_x, N * sizeof(float));
	cudaMemcpy(N_x, x_generados, N * sizeof(float), cudaMemcpyHostToDevice);
	
    cudaMalloc(&N_y, N * sizeof(float));
    
    int block_size = 256;
    int grid_size = (int)ceil((float)N / block_size);

    cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);
    BaryKernel << < grid_size, block_size >> > (X, Y, N_x, devWeights, N_y, N, M);
    cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);

    cout << "[GPU] Duration: " << dt << " [ms]" << endl;

    cudaMemcpy(y_generados, N_y, N * sizeof(float), cudaMemcpyDeviceToHost);

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
    delete weights;
    delete x_generados;
    delete y_generados;
    
    return 0;
}
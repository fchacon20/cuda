#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

using namespace std;

void initialPoints(float *x, float *y, int M, int a, int b) {
	
	for (int i = 1; i <= M; ++i) {
		x[i - 1] = (double)(a + b) / 2 + (double)(((b - a) / 2.0)*cos((2.0*i - 1.0)*M_PI / ((double)2.0*M)));
		y[i - 1] = cos(x[i - 1]);
	}

}

void generateX(float *x_generados, int N) {
	for (int i = 1; i <= N; ++i)
		x_generados[i-1] = ((float) (b-a)/(n)*i);
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
        
        for(int j = 0; j < M; j++){
            num += (devWeights[j]/(N_x[tId] - X[j])) * Y[j];
            den += (devWeights[j]/(N_x[tId] - X[j]));
        }
        N_y[tId] = num / den;
    }
}

__constant__ double constantWeight[30];
__global__ void ConsBaryKernel(const float * __restrict__ X, const float * __restrict__ Y, const float * __restrict__ N_x, float* N_y, int N, int M){

    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    if(tId < N){
        double num = 0;
        double den = 0;

        for(int j = 0; j < M; j++){
            num += (constantWeight[j]/(N_x[tId] - X[j])) * Y[j];
            den += (constantWeight[j]/(N_x[tId] - X[j]));
        }
        N_y[tId] = num / den;
    } 
}
int main(){

    int N = 1000000;	// Cantidad de puntos a interpolar
    int M = 30;			// Cantidad de puntos a utilizar de la funci�n original
	int a = 0;
	int b = 100;

    double * weights = new double[M];
    cudaEvent_t ct1, ct2, ct3, ct4;
    float dt, dt2;

	float *X, *X2, *Y, *Y2;			// Arreglos conteniendo coordenadas X e Y de puntos de la funci�n original
	float *N_x, *N_x2, *N_y, *N_y2;	// Arreglos conteniendo coordenadas X e Y de puntos a interpolar.
									// N_x deben generarse, N_y deben calcularse usando el kernel
    double *devWeights;

	float *x = new float[M]; // coordenadas x de la funci�n
	float *y = new float[M]; // coordenadas y de la funci�n
	float *x_generados = new float[N];
    float *y_generados = new float[N];
	float *y_generados2 = new float[N];


    initialPoints(x, y, M, a, b);
    generateX(x_generados, N);
    calculateWeights(weights, x, M);

    // Saving input
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

	cudaMalloc(&X, M * sizeof(float));
	cudaMalloc(&Y, M * sizeof(float));
	cudaMalloc(&devWeights, M * sizeof(double));
	cudaMalloc(&N_x, N * sizeof(float));
	cudaMalloc(&N_y, N * sizeof(float));

	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);

	cudaMemcpy(X, x, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Y, y, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devWeights, weights, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(N_x, x_generados, N * sizeof(float), cudaMemcpyHostToDevice);

    BaryKernel << < grid_size, block_size >> > (X, Y, N_x, devWeights, N_y, N, M);

	cudaMemcpy(y_generados, N_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);

    cout << "[GPU] Duration: " << dt << " [ms]" << endl;

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
	cudaFree(devWeights);


	cudaMalloc(&X2, M * sizeof(float));
	cudaMalloc(&Y2, M * sizeof(float));
	cudaMalloc(&N_x2, N * sizeof(float));
	cudaMalloc(&N_y2, N * sizeof(float));

	cudaEventCreate(&ct3);
	cudaEventCreate(&ct4);
	cudaEventRecord(ct3);

	cudaMemcpy(X2, x, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Y2, y, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constantWeight, weights, M * sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpy(N_x2, x_generados, N * sizeof(float), cudaMemcpyHostToDevice);


    ConsBaryKernel << < grid_size, block_size >> > (X2, Y2, N_x2, N_y2, N, M);

	cudaMemcpy(y_generados2, N_y2, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(ct4);
	cudaEventSynchronize(ct4);
	cudaEventElapsedTime(&dt2, ct3, ct4);

    cout << "[GPU-Const] Duration: " << dt2 << " [ms]" << endl;

	ofstream outfile3("./output_const.txt");
	for (int i = 0; i < N - 1; ++i)
		outfile3 << x_generados[i] << ",";
	outfile3 << x_generados[N - 1] << "\n";
	for (int i = 0; i < N - 1; ++i)
		outfile3 << y_generados2[i] << ",";
	outfile3 << y_generados2[N - 1] << "\n";
    outfile3.close();
    
    cudaFree(X2);
	cudaFree(Y2);
	cudaFree(N_x2);
	cudaFree(N_y2);
	
	delete x;
    delete y;
    delete weights;
    delete x_generados;
    delete y_generados;
	delete y_generados2;
    
    return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <istream>
#include <iterator>
#include "functions.cuh"
#include <stdio.h>
#include <time.h>
using namespace std;

__global__ void invertColorsKernel(int size, float* R, float* G, float *B) {
	int tId = threadIdx.x + blockIdx.x*blockDim.x;
	if (tId < size) {
		R[tId] = 1 - R[tId];
		G[tId] = 1 - G[tId];
		B[tId] = 1 - B[tId];
	}
}

void invertColorsCPU(int size, float* R, float* G, float *B) {
	for (int i = 0; i < size; ++i) {
		R[i] = 1 - R[i];
		G[i] = 1 - G[i];
		B[i] = 1 - B[i];
	}
}

int main()
{
	ifstream file("C:/Users/felip/Desktop/git/cuda/practico1/practico1/images/img1.txt");
	int N, M;
	string line;
	vector<vector<string>> lines;
	while (getline(file, line)) {
		istringstream iss(line);
		lines.push_back(vector<string>(istream_iterator<string> { iss }, istream_iterator<string>()));
	}
	N = stoi(lines[0][0]);
	M = stoi(lines[0][1]);


	float* R = new float[N*M];
	float* G = new float[N*M];
	float* B = new float[N*M];
	float *devR, *devG, *devB;

	for (int i = 0; i < N*M; ++i) {
		R[i] = stof(lines[1][i]);
		G[i] = stof(lines[2][i]);
		B[i] = stof(lines[3][i]);
	}

	cudaMalloc(&devR, N*M * sizeof(float));
	cudaMalloc(&devG, N*M * sizeof(float));
	cudaMalloc(&devB, N*M * sizeof(float));

	cudaMemcpy(devR, R, N*M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devG, G, N*M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, B, N*M * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t ct1, ct2;
	float dt;
	int grid_size = (int)ceil((float)N*M / 256);
	int block_size(256);
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);
	invertColorsKernel << <grid_size, block_size >> > (N*M, devR, devG, devB);
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	cudaMemcpy(R, devR, N*M * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Tiempo de ejecucion en GPU: " << dt << " [ms]" << endl;

	// Invert colors
	clock_t begin = clock();
	invertColorsCPU(N*M,R,G,B);
	clock_t end = clock();
	double time_spent = (double) (end - begin) * 1000 / CLOCKS_PER_SEC;
	cout << "Tiempo de ejecucion en CPU: " << time_spent << " [ms]" << endl;


	/*
	clock_t begin = clock();
	axialReflection('V', N, M, R, G, B);

	for (int i = 0; i < N*M; ++i)
		cout << R[i] << " ";
	cout << endl;
	for (int i = 0; i < N*M; ++i)
		cout << G[i] << " ";
	cout << endl;
	for (int i = 0; i < N*M; ++i)
		cout << B[i] << " ";
	cout << endl << endl;
	*/

	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

	file.close();
	delete[] R;
	delete[] G;
	delete[] B;

	cudaFree(devR);
	cudaFree(devG);
	cudaFree(devB);
    return 0;
}




#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <istream>
#include <iterator>

using namespace std;

__global__ void thirdCollision(int *devF, int* devF1, int N, int M) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	//if (tId == 0)
	//	printf("devF1[0]: %d\n", devF1[0]);

	if (tId < N*M) {
		int f0, f1, f2, f3;
		f3 = devF[tId] / 8;
		devF[tId] = devF[tId] % 8;
		f2 = devF[tId] / 4;
		devF[tId] = devF[tId] % 4;
		f1 = devF[tId] / 2;
		devF[tId] = devF[tId] % 2;
		f0 = devF[tId];

		// Collisions
		if (tId >= M && tId < (N*M - M) && tId % M && (tId + 1) % M) {
			if (f0 == 1 && f1 == 0 && f2 == 1 && f3 == 0) {
				f0 = 0;
				f2 = 0;
				f1 = 1;
				f3 = 1;
			}
			else if (f0 == 0 && f1 == 1 && f2 == 0 && f3 == 1) {
				f0 = 1;
				f2 = 1;
				f1 = 0;
				f3 = 0;
			}
		}

		//Streaming
		if (f0 == 1) {
			if ((tId + 1) % M == 0)
				atomicAdd(&devF1[tId], 4);
			else
				atomicAdd(&devF1[tId + 1], 1);
		}
		if (f1 == 1) {
			if (tId < M)
				atomicAdd(&devF1[tId], 8);
			else
				atomicAdd(&devF1[tId - M], 2);
		}
		if (f2 == 1) {
			if ((tId % M) == 0)
				atomicAdd(&devF1[tId], 1);
			else
				atomicAdd(&devF1[tId - 1], 4);
		}
		if (f3 == 1) {
			if (tId >= (N*M - M))
				atomicAdd(&devF1[tId], 2);
			else
				atomicAdd(&devF1[tId + M], 8);
		}
	}
}

__global__ void print(int*devF, int N, int M) {
	for (int i = 0; i < N*M; i++)
	{
		printf("%d ", devF[i]);
	}
	printf("\n");
}

__global__ void thirdFinalStep(int* devF, int* devF1, int N, int M) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;

	if (tId < N*M)
		devF[tId] = devF1[tId];
}

void thirdVersion(vector<vector<string>> lines, int* F, int N, int M) {
	for (int i = 0; i < N*M; ++i) {
		F[i] = stoi(lines[1][i]) * 1 +
			stoi(lines[2][i]) * 2 +
			stoi(lines[3][i]) * 4 +
			stoi(lines[4][i]) * 8;
	}
}


int main() {

	ifstream file("./initial.txt");
	clock_t begin = clock();
	int N, M, count = 1;
	string line;
	vector<vector<string>> lines;
	cudaEvent_t ct1, ct2;
	float durationGPU;
	int particles = 0;

	while (getline(file, line)) {
		cout << "Cargando linea " << count << " de 5." << endl;
		istringstream iss(line);
		lines.push_back(vector<string>(istream_iterator<string> { iss }, istream_iterator<string>()));
		count++;
	}

	file.close();

	N = stoi(lines[0][0]);
	M = stoi(lines[0][1]);

	int* F = new int[N*M];
	thirdVersion(lines, F, N, M);


	for (int i = 0; i < N*M; i++)
	{
		int f0, f1, f2, f3;
		f3 = F[i] / 8;
		f2 = (F[i] % 8) / 4;
		f1 = ((F[i] % 8) % 4) / 2;
		f0 = ((F[i] % 8) % 4) % 2;

		//cout << f0 << " " << f1 << " " << f2 << " " << f3 << endl;
		particles += f0 + f1 + f2 + f3;
	}

	cout << "Cantidad de particulas: " << particles << endl;

	int* devF;
	int* devF1;
	cudaMalloc(&devF, N*M * sizeof(int));
	cudaMalloc(&devF1, N*M * sizeof(int));
	cudaMemcpy(devF, F, N*M * sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (int)ceil((float)N*M / blockSize);

	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);
	for (int i = 0; i < 1; i++)
	{
		cudaMemset(devF1, 0, N*M * sizeof(int));
		thirdCollision << <gridSize, blockSize >> > (devF, devF1, N, M);
		cudaDeviceSynchronize();
		thirdFinalStep << <gridSize, blockSize >> > (devF, devF1, N, M);
	}

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&durationGPU, ct1, ct2);

	cudaMemcpy(F, devF, N*M * sizeof(int), cudaMemcpyDeviceToHost);

	particles = 0;
	for (int i = 0; i < N*M; i++)
	{
		int f0, f1, f2, f3;
		f3 = F[i] / 8;
		f2 = (F[i] % 8) / 4;
		f1 = ((F[i] % 8) % 4) / 2;
		f0 = ((F[i] % 8) % 4) % 2;

		//cout << f0 << " " << f1 << " " << f2 << " " << f3 << endl;
		particles += f0 + f1 + f2 + f3;

	}

	cout << "Cantidad de particulas: " << particles << endl;
	cout << "[GPU] Duracion: " << durationGPU << " [ms]" << endl;

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

	cout << "Liberando memoria" << endl;
	delete[] F;
	cudaFree(devF);
	cudaFree(devF1);
	cudaDeviceSynchronize();

	cout << "Programa terminado" << endl;

	return 0;
}

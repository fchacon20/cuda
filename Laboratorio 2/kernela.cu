
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

__global__ void timeStepSoA(int *devF, int *devF1, int N, int M){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;

	if (tId < N*M) {
		int f0, f1, f2, f3, x, y;
		f0 = devF[tId];
		f1 = devF[tId + N*M];
		f2 = devF[tId + N*M*2];  
		f3 = devF[tId + N*M*3];
		x = tId / M;
		y = tId % M;
		// Collisions
		if (f0 == 1 && f1 == 0 && f2 == 1 && f3 == 0) {
			f0 = 0;
			f2 = 0;
			f1 = 1;
			f3 = 1;
		} else if (f0 == 0 && f1 == 1 && f2 == 0 && f3 == 1) {
			f0 = 1;
			f2 = 1;
			f1 = 0;
			f3 = 0;
		}
		//Streaming
		if(f0 == 1){
			if(y == M-1)
				devF1[x*M] = 1;
			else
				devF1[x*M + y + 1] = 1;
		}
		if(f1 == 1){  
			if(x == 0)
				devF1[(N-1)*M + y + N*M] = 1;
			else
				devF1[(x-1)*M + y + N*M] = 1;
		}
		if(f2 == 1){
			if(y == 0)
				devF1[x*M + M-1 + N*M*2] = 1;
			else
				devF1[x*M + y - 1 + N*M*2] = 1;
		}
		if(f3 == 1){
			if(x == N-1)
				devF1[y + N*M*3] = 1;
			else
				devF1[(x+1)*M + y + N*M*3] = 1;
		} 
	}
}

__global__ void timeStepAoS(int *devF, int *devF1, int N, int M){
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tId < N*M){
		int f0, f1, f2, f3, x, y;
		x = tId / M;
		y = tId % M;
		tId = tId * 4;
		f0 = devF[tId];
		f1 = devF[tId + 1];
		f2 = devF[tId + 2];
		f3 = devF[tId + 3];
		
		// Collisions
		if (f0 == 1 && f1 == 0 && f2 == 1 && f3 == 0) {
			f0 = 0;
			f2 = 0;
			f1 = 1;
			f3 = 1;
		} else if (f0 == 0 && f1 == 1 && f2 == 0 && f3 == 1) {
			f0 = 1;
			f2 = 1;
			f1 = 0;
			f3 = 0;
		}
		// Streaming
		if(f0 == 1){
			if(y == M-1)
				devF1[x*M*4] = 1;
			else
				devF1[(x*M + y + 1)*4] = 1;
		}
		if(f1 == 1){  
			if(x == 0)
				devF1[((N-1)*M + y)*4 + 1] = 1;
			else
				devF1[((x-1)*M + y)*4 + 1] = 1;
		}
		if(f2 == 1){
			if(y == 0)
				devF1[(x*M + M-1)*4 + 2] = 1;
			else
				devF1[(x*M + y - 1)*4 + 2] = 1;
		}
		if(f3 == 1){
			if(x == N-1)
				devF1[y*4 + 3] = 1;
			else
				devF1[((x+1)*M + y)*4 + 3] = 1;
		}
	}
}

__global__ void FinalStep (int* devF, int* devF1, int N, int M) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;

	if (tId < N*M*4)
		devF[tId] = devF1[tId];
}

int * SoAGen(vector<vector<string> > lines, int N, int M){
	int * SoA_array = new int [N*M*4];
	int contador = 0;
	for (int i = 1; i <= 4; i++){
		for (int j = 0; j < N*M; j++){
			SoA_array[contador] = stoi(lines[i][j]);
			contador++;
		}
	}
	return SoA_array;
}

int * AoSGen(vector<vector<string> > lines, int N, int M){
	int * AoS_array = new int [N*M*4];
	int contador = 0;
	for (int i = 0; i < N*M; i++){
		for (int j = 1; j <= 4; j++){
			AoS_array[contador] = stoi(lines[j][i]);
			contador++;
		}
	}
	return AoS_array;
}

int main(){

	ifstream file("initial.txt");
	int N, M, count = 1;
	string line;
	vector<vector<string> > lines;
	cudaEvent_t ct1, ct2;
	float durationGPU;
	int particlesAoS = 0;
	int particlesSoA = 0;
	int blockSize = 256;
	int gridSize;
	int * devSoa;
	int * devAos;
	int * devF1;
	int * SoA;
	int * AoS;

	while (getline(file, line)) {
		cout << "Cargando linea " << count << " de 5." << endl;
		istringstream iss(line);
		lines.push_back(vector<string>(istream_iterator<string> { iss }, istream_iterator<string>()));
		count++;
	}

	file.close();

	N = stoi(lines[0][0]);
	M = stoi(lines[0][1]);
	gridSize = (int)ceil((float)N*M*4 / blockSize);

	SoA = SoAGen(lines, N, M);
	AoS = AoSGen(lines, N, M);
	cudaMalloc(&devSoa, N*M*4 * sizeof(int));
	cudaMalloc(&devF1, N*M*4 * sizeof(int));
	cudaMalloc(&devAos, N*M*4 * sizeof(int));
	cudaMemcpy(devSoa, SoA, N*M*4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devAos, AoS, N*M*4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);
	
	for (int i = 0; i < N*M*4; i++){
		particlesSoA += SoA[i];
		particlesAoS += AoS[i];
	}
	cout << "En SoA hay " << particlesSoA << " particulas y en AoS hay " << particlesAoS << " particulas" << endl;
	
	for(int i = 0; i < 1000; i++){
		cudaMemset(devF1, 0, N*M*4 * sizeof(int));
		timeStepSoA << <gridSize, blockSize >> > (devSoa, devF1, N, M);
		cudaDeviceSynchronize();
		FinalStep << <gridSize, blockSize >> > (devSoa, devF1, N, M);
	}

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&durationGPU, ct1, ct2);

	cudaMemcpy(SoA, devSoa, N*M*4 * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Tiempo de ejecucion SoA: " << durationGPU << "[ms]" << endl;
	
	cudaEventRecord(ct1);

	for(int i = 0; i < 1000; i++){
		cudaMemset(devF1, 0, N*M*4 * sizeof(int));
		timeStepAoS << <gridSize, blockSize >> > (devAos, devF1, N, M);
		cudaDeviceSynchronize();
		FinalStep << <gridSize, blockSize >> > (devAos, devF1, N, M);
	}
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&durationGPU, ct1, ct2);

	cudaMemcpy(AoS, devAos, (N*M*4 * sizeof(int)), cudaMemcpyDeviceToHost);

	cout << "Tiempo de ejecucion AoS: " << durationGPU << "[ms]" << endl;

	particlesSoA = 0;
	particlesAoS = 0;
	for (int i = 0; i < N*M*4; i++){
		particlesSoA += SoA[i];
		particlesAoS += AoS[i];
	}
	cout << "En SoA hay " << particlesSoA << " particulas y en AoS hay " << particlesAoS << " particulas" << endl;
	
	cout << "Programa terminado" << endl;

	delete[] AoS;
	delete[] SoA;
	cudaFree(devF1);
	cudaFree(devSoa);
	cudaFree(devAos);

    return 0;
}

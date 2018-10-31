
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

__global__ void thirdCollision(int *devF, int N, int M){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;

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
		if (f0 == 1 && f1 == 0 && f2 == 1 && f3 == 0) {
			f0 = f2 = 0;
			f1 = f3 = 1;
		} else if (f0 == 0 && f1 == 1 && f2 == 0 && f3 == 1) {
			f0 = f2 = 1;
			f1 = f3 = 0;
		}

		//Streaming

	}

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

int* thirdVersion(vector<vector<string>> lines, int N, int M) {
	int* F = new int[N*M];
	for (int i = 0; i < N*M; ++i) {
		F[i] = stoi(lines[1][i]) * 1 +
			stoi(lines[2][i]) * 2 +
			stoi(lines[3][i]) * 4 +
			stoi(lines[4][i]) * 8;
	}
	return F;
}

int main(){

	ifstream file("initial.txt");
	clock_t begin = clock();
	int N, M, count = 1;
	string line;
	vector<vector<string> > lines;
	cudaEvent_t ct1, ct2;
	float durationGPU;

	while (getline(file, line)) {
		cout << "Cargando linea " << count << " de 5." << endl;
		istringstream iss(line);
		lines.push_back(vector<string>(istream_iterator<string> { iss }, istream_iterator<string>()));
		count++;
	}

	file.close();

	N = stoi(lines[0][0]);
	M = stoi(lines[0][1]);

	int * SoA = SoAGen(lines, N, M);
	int * AoS = AoSGen(lines, N, M);
	int* F = thirdVersion(lines, N, M);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

	cout << "Programa terminado" << endl;

	delete[] AoS;
	delete[] SoA;
	delete[] F;

    return 0;
}


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

__global__ void addKernel(int *c, const int *a, const int *b){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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
	clock_t begin = clock();
	int N, M, count = 1;
	string line;
	vector<vector<string> > lines;

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

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

	cout << "Programa terminado" << endl;

	delete[] AoS;
	delete[] SoA;

    return 0;
}

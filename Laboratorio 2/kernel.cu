
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

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(){

	ifstream file("./initial.txt");
	clock_t begin = clock();
	int N, M, count = 1;
	string line;
	vector<vector<string>> lines;

	while (getline(file, line)) {
		cout << "Cargando linea " << count << " de 5." << endl;
		istringstream iss(line);
		lines.push_back(vector<string>(istream_iterator<string> { iss }, istream_iterator<string>()));
	}

	file.close();

	N = stoi(lines[0][0]);
	M = stoi(lines[0][1]);

	int* F0 = new int[N*M];
	int* F1 = new int[N*M];
	int* F2 = new int[N*M];
	int* F3 = new int[N*M];

	for (int i = 0; i < N*M; ++i) {
		F0[i] = stoi(lines[1][i]);
		F0[i] = stoi(lines[2][i]);
		F0[i] = stoi(lines[3][i]);
		F0[i] = stoi(lines[4][i]);
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

	cout << "Programa terminado" << endl;

	delete[] F0;
	delete[] F1;
	delete[] F2;
	delete[] F3;

    return 0;
}

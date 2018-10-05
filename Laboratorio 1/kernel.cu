
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <chrono>

#define E 2.71828182845904523536

using namespace std;

float* eulerMethodCPU(int t0, float y0, float deltaT) {
	int n = int(10 / deltaT);
	float* y = new float[n];
	float sum;
	
	y[0] = y0;
	for (int i = 1; i < n; i++)	{
		sum = 0;
		for (int j = 0; j < i; j++)
			sum = sum + powf(E, -1 *j*deltaT);
		y[i] = y0 + deltaT * sum;
	}
	return y;
}


int main() {
	
	float* y;
	float deltaT[6] = { 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 };
	chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

	for (int i = 0; i < 6; i++) 
		y = eulerMethodCPU(0, -1, deltaT[i]);
	
	chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
	long duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
	cout << "Tiempo de ejecucion en CPU: " << duration << " [ms]" << endl;

	delete[] y;
	return 0;
}
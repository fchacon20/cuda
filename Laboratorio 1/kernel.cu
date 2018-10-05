
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>

#define E 2.71828182845904523536

using namespace std;

float* eulerMethodCPU(int t0, float y0, float deltaT) {
	int n = int(10 / deltaT);
	float* y = new float[n];
	float sum = 0.0;
	
	y[0] = y0;
	for (int i = 1; i < n; i++)	{
		sum = sum + powf(E, -1 *(i-1)*deltaT);
		y[i] = y0 + deltaT * sum;
	}
	return y;
}


int main() {
	
	float* y;
	float deltaT[6] = { 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 };
	clock_t t1;
	clock_t t2;
	long duration;

	ofstream outputFile;
	outputFile.open("euler.txt");

	for (int i = 0; i < 6; i++) {
		t1 = clock();
		y = eulerMethodCPU(0, -1, deltaT[i]);
		t2 = clock();
		duration = 1000*(double)(t2 - t1) / CLOCKS_PER_SEC;
		cout << "[CPU] DeltaT = " << deltaT[i] << ": " << duration << " [ms]" << endl;

		for (int j = 0; j < int(10 / deltaT[i])-1; j++)
			outputFile << y[j] << " ";
		outputFile << y[int(10 / deltaT[i]) - 1] << endl;
	}	

	outputFile.close();
	delete[] y;
	return 0;
}
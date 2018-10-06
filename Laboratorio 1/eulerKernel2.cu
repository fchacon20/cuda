#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>

#define E 2.71828182845904523536

using namespace std;

void actualizadorCPU(float * soluciones, int m, float deltaT, float tI){
    int j;
    for(j = 0; j < m; j++)
        soluciones[j] = soluciones[j] + deltaT * (4 * (tI - deltaT) - soluciones[j] + 3 + j);
}

void secondCPU_GPU(int m){
    int n = 1000;
    long duration;
    float deltaT = 0.001;
    clock_t t1, t2;
    float * soluciones = new float [m];
    int i;
    for(i = 0; i < m; i++)
        soluciones[i] = i;
    t1 = clock();
    for(i = 1; i < n; i++)
        actualizadorCPU(soluciones, m, deltaT, deltaT*i);
    t2 = clock();
    duration = 1000*(double)(t2 - t1) / CLOCKS_PER_SEC;
    cout << "[CPU] Tamaño m = " << m << ": " << duration << " [ms]" << endl;
    
    delete soluciones;
}

int main(){

    secondCPU_GPU(10000);
    secondCPU_GPU(100000);
    secondCPU_GPU(1000000);
    secondCPU_GPU(10000000);
    secondCPU_GPU(100000000);

    return 0;
}
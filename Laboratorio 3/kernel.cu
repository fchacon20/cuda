
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// consideraciones

/*
Se trabajar´a con una matriz cuadrada de 10^8 elementos, i.e. N = M = 10^4
El tamano de un bloque de hebras siempre sera 256.

Los valores especificados en los dos puntos anteriores pueden ser declarados como constantes en
tiempo de compilacion.

La matriz A y el vector x pueden ser inicializados con los valores que guste mientras no contengan
valores nulos (0). Como consejo, se le recomienda que todos los valores sean 1 para entonces poder
comprobar si el resultado es correcto cuando todos los valores en b sean 10^4.
*/

__global__ void kernelA(int *A, int*x, int*b, int N) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	
	int i = tId % N;
	int j = tId / N;
	if (i < N && j < N) {
		atomicAdd(A[i][j]*x[j], b[i]);
	}
	/*
	Este kernel utiliza N x N = 10^8 hebras. Cada hebra esta asociada a un elemento a_i,j de la matriz A
	multiplicándolo por el valor x_j correspondiente y sumando este resultado al elemento en la i-ésima
	posición del vector b.
	*/
}

__global__ void kernelx(int *A, int*x, int*b, int N) {
	/*
	Este kernel utiliza N = 10^4 hebras. Cada hebra esta asociada a un elemento x_j del vector x,
	sumando a cada uno de los N valores b_i	, la multiplicacion de dicho x_j 
	por el correspondiente elemento a_i,j
	*/
}


__global__ void kernelb(int *A, int*x, int*b, int N) {


}

__global__ void kernelRed(int *A, int*x, int*b, int N) {
}

__global__ void kernelSM(int *A, int*x, int*b, int N) {
}

__global__ void kernelCM(int *A, int*b, int N) {
}
int main() {
	return 0;
}

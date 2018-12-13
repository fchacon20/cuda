#include <iostream>
#include <cmath>

using namespace std;

void initialPoints(double *x, double *y, int n){
    for (int i = 0; i < n; ++i) {
        x[i] = i;
        y[i] = 3*x[i]+(double)pow(x[i],2)+1;
    }
}

int main() {

    int n = 30;
    double* x = new double[n];
    double* y = new double[n];
    double* weights = new double[n];

    initialPoints(x, y, n);

    clock_t begin = clock();

    double point;
    double num = 0, den = 0;
    double result = 0;

    for (int i = 0; i < n; i++){
        weights[i] = 1;
        for (int j = 0; j < n; j++){
            if(i != j)
                weights[i] *= (x[i] - x[j]);
        }
        weights[i] = (double)pow(weights[i], -1);
    }

    for (int i = 1; i <= 1000000; i++) {
        point = (double) 0.00003*i;
        for(int j = 0; j < n; j++){
            num += (weights[j]/(point - x[j])) * y[j];
            den += (weights[j]/(point - x[j]));
        }
        result = num / den;
        num = den = 0;
    }

    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

    return 0;
}
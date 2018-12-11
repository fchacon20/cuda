#include <iostream>
#include <cmath>

using namespace std;

void initialPoints(float *x, float *y, int n){
    for (int i = 0; i < n; ++i) {
        x[i] = i;
        y[i] = 3*x[i]+(float)pow(x[i],2)+1;
    }
}

int main() {

    int n = 30;
    float* x = new float[n];
    float* y = new float[n];

    initialPoints(x, y, n);

    clock_t begin = clock();

    float point;
    float num = 1, den = 1;
    float frac, result = 0;

    for (int k = 1; k <= 1000000; ++k) {
        point = (float) 0.00003*k;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    num *= (point - x[j]);
                    den *= (x[i] - x[j]);
                }
            }
            frac = num / den;
            result += y[i] * frac;
            num = den = 1;
        }
        //cout << result << endl;
        result = 0;
    }

    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

    return 0;
}
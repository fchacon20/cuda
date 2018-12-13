#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

void initialPoints(float *x, float *y, int n){
    for (int i = 0; i < n; ++i) {
        x[i] = i;
        y[i] = 3*x[i]+(float)pow(x[i],2)+1;
    }
}

int main() {

    int n = 30;
    int m = 1000000;
    float* x = new float[n];
    float* y = new float[n];
    float* xn = new float[m];
    float* yn = new float[m];

    initialPoints(x, y, n);

    // Saving input
    ofstream outfile("./initialPoints.txt");
    for (int i = 0; i < n-1; ++i)
        outfile << x[i] << ",";
    outfile << x[n-1] << "\n";
    for (int i = 0; i < n-1; ++i)
        outfile << y[i] << ",";
    outfile << y[n-1] << "\n";
    outfile.close();

    clock_t begin = clock();

    float num = 1, den = 1;
    float frac, result = 0;

    for (int k = 1; k <= m; ++k) {
        xn[k-1] = (float) 0.00003*k;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    num *= (xn[k-1] - x[j]);
                    den *= (x[i] - x[j]);
                }
            }
            frac = num / den;
            result += y[i] * frac;
            num = den = 1;
        }
        yn[k-1] = result;
        result = 0;
    }

    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

    // Saving output
    ofstream outfile2("./output.txt");
    for (int i = 0; i < m-1; ++i)
        outfile2 << xn[i] << ",";
    outfile2 << xn[m-1] << "\n";
    for (int i = 0; i < m-1; ++i)
        outfile2 << yn[i] << ",";
    outfile2 << yn[m-1] << "\n";
    outfile2.close();

    return 0;
}

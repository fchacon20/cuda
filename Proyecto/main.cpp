#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

void initialPoints(double *x, double *y, int n, int a, int b){
    for (int i = 1; i <= n; ++i) {
        x[i-1] = (double)(a+b)/2 + (double)(((b-a)/2.0)*cos((2.0*i-1.0)*M_PI/((double)2.0*n)));
        //y[i-1] = 3*x[i-1]+(double)pow(x[i-1],2)+1;
        y[i-1] = cos(x[i-1]);
    }
}

int main() {

    int n = 30;
    int m = 1000000;
    int a = 0;
    int b = 100;
    double* x = new double[n];
    double* y = new double[n];
    double* xn = new double[m];
    double* yn = new double[m];

    initialPoints(x, y, n, a, b);

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

    double num = 1, den = 1;
    double frac, result = 0;

    for (int k = 1; k <= m; ++k) {
        xn[k-1] = ((double)(b-a)/(m))*k;
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

    delete x;
    delete y;
    delete xn;
    delete yn;

    return 0;
}
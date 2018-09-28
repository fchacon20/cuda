#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <istream>
#include <iterator>
#include "functions.h"

using namespace std;

int main() {
    ifstream file ("/home/fchacon/CLionProjects/cuda1/img1.txt");
    clock_t begin = clock();
    int N, M;
    string line;
    vector<vector<string>> lines;

    while(getline(file, line)) {
        istringstream iss(line);
        lines.push_back(vector<string> (istream_iterator<string> { iss }, istream_iterator<string>()));
    }

    N = stoi(lines[0][0]);
    M = stoi(lines[0][1]);

    float* R = NULL;
    float* G = NULL;
    float* B = NULL;

    R = new float[N*M];
    G = new float[N*M];
    B = new float[N*M];

    for (int i = 0; i < N*M; ++i) {
        R[i] = stof(lines[1][i]);
        G[i] = stof(lines[2][i]);
        B[i] = stof(lines[3][i]);
    }

    /*
    // Invert colors
    invertColors(N*M,R,G,B);

    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;
    */

    axialReflection('V',N, M, R, G, B);

    for (int i = 0; i < N*M; ++i)
        cout << R[i] << " ";
    cout << endl;
    for (int i = 0; i < N*M; ++i)
        cout << G[i] << " ";
    cout << endl;
    for (int i = 0; i < N*M; ++i)
        cout << B[i] << " ";
    cout << endl << endl;

    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    cout << "Tiempo de ejecucion: " << time_spent << " segundos" << endl;

    file.close();
    delete [] R;
    delete [] G;
    delete [] B;

    return 0;
}
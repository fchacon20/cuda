//
// Created by fchacon on 27-09-18.
//

void invertColors(int size, float* R, float* G, float *B){
    // Invertir colores
    for (int i = 0; i < size; ++i) {
        R[i] = 1 - R[i];
        G[i] = 1 - G[i];
        B[i] = 1 - B[i];
    }
}

void axialReflection(char axis, int N, int M, float* R, float* G, float* B){
    float aux;
    if(axis == 'V'){
        for (int i = 0; i < N/2; ++i) {
            for (int j = 0; j < M; ++j) {
                aux        = R[j + i*N];
                R[j + i*N] = R[N-1-j+i*N];
                R[N-1-j+i*N] = aux;
            }
        }
        for (int i = 0; i < N/2; ++i) {
            for (int j = 0; j < M; ++j) {
                aux        = G[j + i*N];
                G[j + i*N] = G[N-1-j+i*N];
                G[N-1-j+i*N] = aux;
            }
        }
        for (int i = 0; i < N/2; ++i) {
            for (int j = 0; j < M; ++j) {
                aux        = B[j + i*N];
                B[j + i*N] = B[N-1-j+i*N];
                B[N-1-j+i*N] = aux;
            }
        }
    }
}
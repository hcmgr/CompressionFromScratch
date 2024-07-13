#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void dct(float **DCTMatrix, float **Matrix, int N, int M);
void write_mat(FILE *fp, float **testRes, int N, int M);
void idct(float **Matrix, float **DCTMatrix, int N, int M);
float **calloc_mat(int dimX, int dimY);
void free_mat(float **p);


float **calloc_mat(int dimX, int dimY){
    float **m = calloc(dimX, sizeof(float*));
    float *p = calloc(dimX*dimY, sizeof(float));
    int i;
    for(i=0; i <dimX;i++){
    m[i] = &p[i*dimY];

    }
   return m;
}

void free_mat(float **m){
  free(m[0]);
  free(m);
}

void write_mat(FILE *fp, float **m, int N, int M){

   int i, j;
   for(i =0; i< N; i++){
    fprintf(fp, "%f", m[i][0]);
    for(j = 1; j < M; j++){
       fprintf(fp, "\t%f", m[i][j]);
        }   
    fprintf(fp, "\n");
   }
   fprintf(fp, "\n");
}

void dct(float **DCTMatrix, float **Matrix, int N, int M){

    int i, j, u, v;
    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
        DCTMatrix[u][v] = 0;
            for (i = 0; i < N; i++) {
                for (j = 0; j < M; j++) {
                    DCTMatrix[u][v] += Matrix[i][j] * cos(M_PI/((float)N)*(i+1./2.)*u)*cos(M_PI/((float)M)*(j+1./2.)*v);
                }               
            }
        }
    }  
 }

void idct(float **Matrix, float **DCTMatrix, int N, int M){
    int i, j, u, v;

    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
          Matrix[u][v] = 1/4.*DCTMatrix[0][0];
          for(i = 1; i < N; i++){
          Matrix[u][v] += 1/2.*DCTMatrix[i][0];
           }
           for(j = 1; j < M; j++){
          Matrix[u][v] += 1/2.*DCTMatrix[0][j];
           }

           for (i = 1; i < N; i++) {
                for (j = 1; j < M; j++) {
                    Matrix[u][v] += DCTMatrix[i][j] * cos(M_PI/((float)N)*(u+1./2.)*i)*cos(M_PI/((float)M)*(v+1./2.)*j);
                }               
            }
        Matrix[u][v] *= 2./((float)N)*2./((float)M);
        }
    }  
 }



int main() {

   float    
    testBlockA[8][8] = { {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255},
                         {255, 255, 255, 255, 255, 255, 255, 255} },

    testBlockB[8][8] = {{255, 0, 255, 0, 255, 0, 255, 0},
                        {0, 255, 0, 255, 0, 255, 0, 255},
                        {255, 0, 255, 0, 255, 0, 255, 0},
                        {0, 255, 0, 255, 0, 255, 0, 255},
                        {255, 0, 255, 0, 255, 0, 255, 0},
                        {0, 255, 0, 255, 0, 255, 0, 255},
                        {255, 0, 255, 0, 255, 0, 255, 0},
                        {0, 255, 0, 255, 0, 255, 0, 255} },


    testBlockC[8][8] =  {{16, 11, 10, 16, 24, 40, 51, 61},
                        {12, 12, 14, 19, 26, 58, 60, 55},
                        {14, 13, 16, 24, 40, 57, 69, 56},
                        {14, 17, 22, 29, 51, 87, 80, 62},
                        {18, 22, 37, 56, 68, 109, 103, 77},
                        {24, 35, 55, 64, 81, 104, 113, 92},
                        {49, 64, 78, 87, 103, 121, 120, 101},
                        {72, 92, 95, 98, 112, 100, 103, 99} };

    FILE * fp = fopen("mydata.csv", "w");
    int dimX = 8, dimY = 8;
    int i, j;

    float **testBlock = calloc_mat(dimX, dimY);
    float **testDCT = calloc_mat(dimX, dimY);
    float **testiDCT = calloc_mat(dimX, dimY);

    for(i = 0; i<dimX; i++){
      for(j = 0; j<dimY; j++){
        testBlock[i][j] = testBlockC[i][j];
      }
    }

    write_mat(fp, testBlock, dimX, dimY);

    dct(testDCT, testBlock, dimX, dimY);
    write_mat(fp, testDCT, dimX, dimY);

    idct(testiDCT, testDCT, dimX, dimY);
    write_mat(fp, testiDCT, dimX, dimY);

    fclose(fp);
    free_mat(testBlock);
    free_mat(testDCT);
    free_mat(testiDCT);

    return 0;
}
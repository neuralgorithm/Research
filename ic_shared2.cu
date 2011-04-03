/*
  Sample Implementation of
    Yamazaki and Tanaka (2005).
    Neural Modeling of an Internal Clock.
    Neural Computation 17:1032--1058.
  using only global memory of CUDA.

  Licensed under Creative Commons Attribution License (CC-BY)
    http://creativecommons.org/licenses/by/3.0/
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define N 1024 // To be 2^k
#define T 1000
#define Pr 0.5
#define I 1.0
#define Kappa 2.0
#define Tau 100.0

#define BLOCK_SIZE 512

float *z, *u, *result;
int *w;

void initialize()
{
  int i, j, k;

  w = (int *)malloc(N*N*sizeof(int));
  z = (float *)malloc(N*sizeof(float));
  u = (float *)malloc(N*sizeof(float));
  result = (float *)malloc(T*N*sizeof(float));

  for(i = 0; i < N; i++){
    z[i] = 0;
    u[i] = I;
  }

  srand(23);
  for(i = 0; i < N; i++){
    k = 0;
    for(j = 0; j < N; j++){
      if ((float)rand()/(float)RAND_MAX < Pr){
	w[k+N*i] = j;
	k++;
      }
    }
    w[k+N*i] = -1;
  }
}

void finalize()
{
  free(w);
  free(z);
  free(u);
  free(result);
}

__global__ void Kernel(const int *w, float *z, float *u, float *result, const float decay, const int t)
{
  int i, j, k, n;
  float r, ux;
  __shared__ float zsh[N];

  i = threadIdx.x;
  n = blockIdx.x*BLOCK_SIZE + i;

  zsh[2*i] = z[2*i];
  zsh[2*i+1] = z[2*i+1];
  __syncthreads();

  r = 0;
  for(k = 0; w[k+N*n] != -1; k++){
    j = w[k+N*n];
    r += zsh[j];
  }
  ux = decay*u[n] + (1 - decay)*I - Kappa*r/N;
  if (ux > 0){
    z[n] = ux;
  }else{
    z[n] = 0;
  }
  result[n+N*t] = z[n];
  u[n] = ux;
  __syncthreads();
}

void loop()
{
  float *zd, *ud, *resultd;
  int *wd;
  float decay;
  cudaError_t stat;
  int t;

  decay = exp(-1.0/Tau);

  cudaMalloc((void**)&wd, N*N*sizeof(int));
  cudaMalloc((void**)&zd, N*sizeof(float));
  cudaMalloc((void**)&ud, N*sizeof(float));
  cudaMalloc((void**)&resultd, N*T*sizeof(float));
  cudaMemcpy(wd, w, N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(zd, z, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ud, u, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE);

  for(t = 0; t < T; t++){
    Kernel<<<dimGrid,dimBlock>>>(wd, zd, ud, resultd, decay, t);
  }

  stat = cudaMemcpy(result, resultd, N*T*sizeof(float), cudaMemcpyDeviceToHost);
  if (stat != cudaSuccess){
    puts("error");
  }

  cudaFree(wd);
  cudaFree(zd);
  cudaFree(ud);
  cudaFree(resultd);
}

void output(char *prefix)
{
  FILE *f;
  int t, i;
  char fn[1024];

  sprintf(fn, "%s.r", prefix);
  f = fopen(fn, "w");
  for(t = 0; t < T; t++){
    for(i = 0; i < N; i++){
      if (result[i+N*t] > 0){
	fprintf(f, "%d %d\n", t, i);
      }
    }
  }

  fclose(f);
}

int main(int argc, char *argv[])
{
  char *prefix;

  if (argc < 2){
    fprintf(stderr, "%s <prefix>\n", argv[0]);
    exit(1);
  }
  prefix = argv[1];

  initialize();
  loop();
  output(prefix);
  finalize();

  return 0;
}

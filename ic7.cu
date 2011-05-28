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

#define N 1024 //2048 // To be 2^k
#define T 1000
#define Pr 0.5
#define I 1.0
#define Kappa 2.0
#define Tau 100.0

#define BLOCK_SIZE 64
#define INNER_LOOP 1000

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

__global__ void Kernel(const int *w, float *z, float *u, float *result, const float decay, int t, int *global_sync)
{
  int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  int j, k, s;
  float r;

  volatile float *vz;
  float uLocal, zLocal;
  float resultLocal[INNER_LOOP];
  int wLocal[N];
  float decayLocal;

  vz = z;
  uLocal = u[i];
  for(j = 0; j < N; j++){
    wLocal[j] = w[i*N+j];
  }
  decayLocal = decay;

  for(s = 0; s < INNER_LOOP; s++){
    r = 0;
    for(k = 0; wLocal[k] != -1; k++){
      j = wLocal[k];
      r += vz[j];
    }
    uLocal = decayLocal*uLocal + (1 - decayLocal)*I - Kappa*r/N;

    if (uLocal > 0){
      zLocal = uLocal;
    }else{
      zLocal = 0;
    }
    resultLocal[s] = zLocal;
    vz[i] = zLocal;

    if (threadIdx.x == 0){
      atomicAdd(global_sync, 1);
      if (blockIdx.x == 0){
	while(atomicAdd(global_sync,0) < N/BLOCK_SIZE);
	atomicExch(global_sync,0);
      }else{
	while(atomicAdd(global_sync,0) > 0);
      }
    }
    __syncthreads();
  }
  u[i] = uLocal;
  for(s = 0; s < INNER_LOOP; s++){
    result[i*T+t*INNER_LOOP+s] = resultLocal[s];
  }
}

void loop()
{
  float *zd, *ud, *resultd;
  int *wd;
  float decay;
  cudaError_t stat;
  int t;
  int *global_syncd;

  decay = exp(-1.0/Tau);

  cudaMalloc((void**)&wd, N*N*sizeof(int));
  cudaMalloc((void**)&zd, N*sizeof(float));
  cudaMalloc((void**)&ud, N*sizeof(float));
  cudaMalloc((void**)&resultd, N*T*sizeof(float));
  cudaMemcpy(wd, w, N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(zd, z, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ud, u, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&global_syncd, sizeof(int));

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE);

  for(t = 0; t < T/INNER_LOOP; t++){
    Kernel<<<dimGrid,dimBlock>>>(wd, zd, ud, resultd, decay, t, global_syncd);
  }
  stat = cudaMemcpy(result, resultd, N*T*sizeof(float), cudaMemcpyDeviceToHost);
  if (stat != cudaSuccess){
    puts("error");
  }

  cudaFree(wd);
  cudaFree(zd);
  cudaFree(ud);
  cudaFree(resultd);
  cudaFree(global_syncd);
}

void output(char *prefix)
{
  FILE *f;
  int t, i;
  char fn[1024];

  sprintf(fn, "%s.r", prefix);
  f = fopen(fn, "w");
  for(i = 0; i < N; i++){
    for(t = 0; t < T; t++){
      if (result[t+T*i] > 0){
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

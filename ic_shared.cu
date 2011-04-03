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
//int *w;
int *w11, *w12, *w21, *w22;

void initialize()
{
  int i, j, k;

  //w = (int *)malloc(N*N*sizeof(int));
  w11 = (int *)malloc((N/2)*(N/2)*sizeof(int));
  w12 = (int *)malloc((N/2)*(N/2)*sizeof(int));
  w21 = (int *)malloc((N/2)*(N/2)*sizeof(int));
  w22 = (int *)malloc((N/2)*(N/2)*sizeof(int));

  z = (float *)malloc(N*sizeof(float));
  u = (float *)malloc(N*sizeof(float));
  result = (float *)malloc(T*N*sizeof(float));

  for(i = 0; i < N; i++){
    z[i] = 0;
    u[i] = I;
  }

  srand(23);
  /*
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
  */
  for(i = 0; i < N/2; i++){
    k = 0;
    for(j = 0; j < N/2; j++){
      if ((float)rand()/(float)RAND_MAX < Pr){
	w11[k+(N/2)*i] = j;
	k++;
      }
    }
    w11[k+(N/2)*i] = -1;
    w11[(N/2)-1+(N/2)*i] = -1;
  }
  for(i = 0; i < N/2; i++){
    k = 0;
    for(j = N/2; j < N; j++){
      if ((float)rand()/(float)RAND_MAX < Pr){
	w12[k+(N/2)*i] = j-N/2;
	k++;
      }
    }
    w12[k+(N/2)*i] = -1;
    w12[(N/2)-1+(N/2)*i] = -1;
  }
  for(i = N/2; i < N; i++){
    k = 0;
    for(j = 0; j < N/2; j++){
      if ((float)rand()/(float)RAND_MAX < Pr){
	w21[k+(N/2)*(i-N/2)] = j;
	k++;
      }
    }
    w21[k+(N/2)*(i-N/2)] = -1;
    w21[(N/2)-1+(N/2)*(i-N/2)] = -1;
  }
  for(i = N/2; i < N; i++){
    k = 0;
    for(j = N/2; j < N; j++){
      if ((float)rand()/(float)RAND_MAX < Pr){
	w22[k+(N/2)*(i-N/2)] = j-N/2;
	k++;
      }
    }
    w22[k+(N/2)*(i-N/2)] = -1;
    w22[(N/2)-1+(N/2)*(i-N/2)] = -1;
  }

}

void finalize()
{
  //free(w);
  free(w11);
  free(w12);
  free(w21);
  free(w22);
  free(z);
  free(u);
  free(result);
}

__global__ void Kernel(const int *w11, const int *w12, const int *w21, const int *w22, float *z, float *u, float *result, const float decay, const int t)
{
  int i, j, k;
  float r;

  __shared__ float zsh[N/2];

  i = threadIdx.x;

  if (blockIdx.x == 0){ // i = 0...N/2
    // w11
    zsh[i] = z[i];
    __syncthreads();
    r = 0;
    for(k = 0; w11[k+(N/2)*i] != -1; k++){
      j = w11[k+(N/2)*i];
      r += zsh[j];
    }
    u[i] = decay*u[i] + (1 - decay)*I - Kappa*r/N;
    __syncthreads();

    // w12
    zsh[i] = z[i+N/2];
    __syncthreads();
    r = 0;
    for(k = 0; w12[k+(N/2)*i] != -1; k++){
      j = w12[k+(N/2)*i];
      r += zsh[j];
    }
    u[i] += - Kappa*r/N;
    __syncthreads();

    if (u[i] > 0){
      z[i] = u[i];
    }else{
      z[i] = 0;
    }
    result[i+N*t] = z[i];
  }else{ // i = N/2...N
    // w21
    zsh[i] = z[i];
    __syncthreads();
    r = 0;
    for(k = 0; w21[k+(N/2)*i] != -1; k++){
      j = w21[k+(N/2)*i];
      r += zsh[j];
    }
    u[i+N/2] = decay*u[i+N/2] + (1 - decay)*I - Kappa*r/N;
    __syncthreads();

    // w22
    zsh[i] = z[i+N/2];
    __syncthreads();
    r = 0;
    for(k = 0; w22[k+(N/2)*i] != -1; k++){
      j = w22[k+(N/2)*i];
      r += zsh[j];
    }
    u[i+N/2] += - Kappa*r/N;
    __syncthreads();

    if (u[i+N/2] > 0){
      z[i+N/2] = u[i+N/2];
    }else{
      z[i+N/2] = 0;
    }
    result[(i+N/2)+N*t] = z[i+N/2];
  }
}

void loop()
{
  float *zd, *ud, *resultd;
  //int *wd;
  int *w11d, *w12d, *w21d, *w22d;
  float decay;
  cudaError_t stat;
  int t;

  decay = exp(-1.0/Tau);

  //cudaMalloc((void**)&wd, N*N*sizeof(int));
  cudaMalloc((void**)&w11d, (N/2)*(N/2)*sizeof(int));
  cudaMalloc((void**)&w12d, (N/2)*(N/2)*sizeof(int));
  cudaMalloc((void**)&w21d, (N/2)*(N/2)*sizeof(int));
  cudaMalloc((void**)&w22d, (N/2)*(N/2)*sizeof(int));
  cudaMalloc((void**)&zd, N*sizeof(float));
  cudaMalloc((void**)&ud, N*sizeof(float));
  cudaMalloc((void**)&resultd, N*T*sizeof(float));
  //cudaMemcpy(wd, w, N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(w11d, w11, (N/2)*(N/2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(w12d, w12, (N/2)*(N/2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(w21d, w21, (N/2)*(N/2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(w22d, w22, (N/2)*(N/2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(zd, z, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ud, u, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE);

  for(t = 0; t < T; t++){
    //Kernel<<<dimGrid,dimBlock>>>(wd, zd, ud, resultd, decay, t);
    Kernel<<<dimGrid,dimBlock>>>(w11d, w12d, w21d, w22d, zd, ud, resultd, decay, t);
  }

  stat = cudaMemcpy(result, resultd, N*T*sizeof(float), cudaMemcpyDeviceToHost);
  if (stat != cudaSuccess){
    puts("error");
  }

  //cudaFree(wd);
  cudaFree(w11d);
  cudaFree(w12d);
  cudaFree(w21d);
  cudaFree(w22d);
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

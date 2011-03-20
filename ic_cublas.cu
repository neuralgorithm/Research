/*
  Sample Implementation of
    Yamazaki and Tanaka (2005).
    Neural Modeling of an Internal Clock.
    Neural Computation 17:1032--1058
  using CUBLAS.

  Licensed under Creative Commons Attribution License (CC-BY)
    http://creativecommons.org/licenses/by/3.0/
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cublas.h>

#define N 1000
#define T 1000
#define Pr 0.5
#define I 1.0
#define Kappa 2.0
#define Tau 100.0

float *w, *z, *u, *result, *n;

//extern void init_genrand(unsigned long);
//extern double genrand_real2(void);

float *devPtrW, *devPtrZ, *devPtrU, *devPtrN;
cublasStatus stat;

void initialize()
{
  int i, j;

  w = (float *)malloc(N*N*sizeof(float));
  z = (float *)malloc(N*sizeof(float));
  u = (float *)malloc(N*sizeof(float));
  result = (float *)malloc(T*N*sizeof(float));
  n = (float *)malloc(N*sizeof(float));

  for(i = 0; i < N; i++){
    z[i] = 0;
    u[i] = I;
    n[i] = 1;
  }

  srand(23);
  //init_genrand(23L);

  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      if (i == j){
	w[j+N*i] = 0;
      }else{
	//if (genrand_real2() < Pr){
	if ((float)rand()/(float)RAND_MAX < Pr){
	  w[j+N*i] = 1;
	}else{
	  w[j+N*i] = 0;
	}
      }
    }
  }

  cublasInit();

  stat = cublasAlloc(N*N, sizeof(*w), (void **)&devPtrW);
  stat = cublasSetMatrix(N, N, sizeof(*w), w, N, devPtrW, N);
  stat = cublasAlloc(N, sizeof(*z), (void **)&devPtrZ);
  stat = cublasSetVector(N, sizeof(*z), z, 1, devPtrZ, 1);
  stat = cublasAlloc(N, sizeof(*u), (void **)&devPtrU);
  stat = cublasSetVector(N, sizeof(*u), u, 1, devPtrU, 1);
  stat = cublasAlloc(N, sizeof(*n), (void **)&devPtrN);
  stat = cublasSetVector(N, sizeof(*n), n, 1, devPtrN, 1);

}

void finalize()
{
  free(w);
  free(z);
  free(u);
  free(result);
  free(n);

  cublasFree(devPtrW);
  cublasFree(devPtrZ);
  cublasFree(devPtrU);
  cublasFree(devPtrN);
  cublasShutdown();
}

void loop()
{
  int t, i;
  // int j;
  // float r;
  float decay = exp(-1.0/Tau);

  for(t = 0; t < T; t++){
    /*
    for(i = 0; i < N; i++){
      r = 0;
      for(j = 0; j < N; j++){
	r += w[j+N*i]*z[j];
      }
      u[i] = decay*u[i] + (1 - decay)*I - Kappa*r/N;
    }
    */
    // u = -(Kappa/N) * W * z + (decay) * u
    cublasSgemv('n', N, N, -Kappa/N, devPtrW, N, devPtrZ, 1, decay, devPtrU, 1);
    // u = (1 - decay)*I + u
    cublasSaxpy(N, (1-decay)*I, devPtrN, 1, devPtrU, 1);

    cublasGetVector(N, sizeof(*u), devPtrU, 1, u, 1);
    for(i = 0; i < N; i++){
      if (u[i] > 0){
	z[i] = u[i];
      }else{
	z[i] = 0;
      }
      result[i+N*t] = z[i];
    }
    cublasSetVector(N, sizeof(*z), z, 1, devPtrZ, 1);
  }
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

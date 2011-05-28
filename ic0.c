/*
  Sample Implementation of
    Yamazaki and Tanaka (2005).
    Neural Modeling of an Internal Clock.
    Neural Computation 17:1032--1058.

  Licensed under Creative Commons Attribution License (CC-BY)
    http://creativecommons.org/licenses/by/3.0/
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define N 1024
#define T 1000
#define Pr 0.5
#define I 1.0
#define Kappa 2.0
#define Tau 100.0

double *z, *u, *result;
int *w;

//extern void init_genrand(unsigned long);
//extern double genrand_real2(void);

void initialize()
{
  int i, j, k;

  w = (int *)malloc(N*N*sizeof(int));
  z = (double *)malloc(N*sizeof(double));
  u = (double *)malloc(N*sizeof(double));
  result = (double *)malloc(T*N*sizeof(double));

  for(i = 0; i < N; i++){
    z[i] = 0;
    u[i] = I;
  }

  srand(23);
  //init_genrand(23L);

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

void loop()
{
  int t, i, j, k;
  double r, decay = exp(-1.0/Tau);

  for(t = 0; t < T; t++){
    for(i = 0; i < N; i++){
      r = 0;
      for(k = 0; w[k+N*i] != -1; k++){
	j = w[k+N*i];
	r += z[j];
      }
      u[i] = decay*u[i] + (1 - decay)*I - Kappa*r/N;
    }
    for(i = 0; i < N; i++){
      if (u[i] > 0){
	z[i] = u[i];
      }else{
	z[i] = 0;
      }
      result[i+N*t] = z[i];
    }
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

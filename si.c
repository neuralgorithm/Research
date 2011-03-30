#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define N 1000
#define T 1000

void initialize(const char *filename, float *z)
{
  FILE *file;
  int t, n;
  char buf[1024];

  file = fopen(filename, "r");
  for(t = 0; t < T; t++){
    for(n = 0; n < N; n++){
      fgets(buf, 1024, file);
      z[t*N+n] = atof(buf);
    }
  }
}

void similarity_index(const float *z, float *si)
{
  int t, n, t1, t2;
  float r, denom[T];

  for(t = 0; t < T; t++){
    r = 0;
    for(n = 0; n < N; n++){
      r += z[t*N+n]*z[t*N+n];
    }
    denom[t] = sqrt(r);
  }
  for(t = T; t < T; t++){
    denom[t] = 0;
  }

  for(t1 = 0; t1 < T; t1++){
    for(t2 = 0; t2 < T; t2++){
      r = 0;
      for(n = 0; n < N; n++){
	r += z[t1*N+n]*z[t2*N+n];
      }
      si[t1*T+t2] = r;
      if (denom[t1] < 1e-6 || denom[t2] < 1e-6){
	si[t1*T+t2] = 0;
      }else{
	si[t1*T+t2] /= (denom[t1]*denom[t2]);
      }
    }
  }
}
void output(char *filename, float *si)
{
  FILE *file;
  int t1, t2;

  file = fopen(filename, "w");
  for(t1 = 0; t1 < T; t1++){
    for(t2 = 0; t2 < T; t2++){
      fprintf(file, "%d %d %f\n", t1, t2, si[t1*T+t2]);
    }
    fprintf(file, "\n");
  }
}

int main(int argc, char *argv[])
{
  float *z, *si;

  if (argc < 3){
    fprintf(stderr, "usage: %s <input> <output>\n", argv[0]);
    exit(1);
  }

  z = (float *)malloc(T*N*sizeof(float));
  si = (float *)malloc(T*N*sizeof(float));

  initialize(argv[1], z);
  similarity_index(z, si);
  output(argv[2], si);

  free(z);
  free(si);

  return 0;
}

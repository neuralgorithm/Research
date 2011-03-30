#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define N 1000
#define T 1000
#define S 1024

#define BLOCK_SIZE 16

void initialize(const char *filename, float *z)
{
  FILE *file;
  int t, n;
  char buf[1024];

  file = fopen(filename, "r");
  for(t = 0; t < S; t++){
    for(n = 0; n < S; n++){
      z[t*S+n] = 0;
    }
  }
  for(t = 0; t < T; t++){
    for(n = 0; n < N; n++){
      fgets(buf, 1024, file);
      z[t*S+n] = atof(buf);
    }
  }
}

__global__ void Kernel(const float *z, const float *denom, float *si)
{
  int t1 = blockIdx.x*(BLOCK_SIZE) + threadIdx.x;
  int t2 = blockIdx.y*(BLOCK_SIZE) + threadIdx.y;
  int n;
  float r;

  r = 0;
  for(n = 0; n < S; n++){
    r += z[t1*S+n]*z[t2*S+n];
  }
  si[t1*S+t2] = r;

  if (denom[t1] < 1e-6 || denom[t2] < 1e-6){
    si[t1*S+t2] = 0;
  }else{
    si[t1*S+t2] /= (denom[t1]*denom[t2]);
  }

}

void similarity_index(const float *z, float *si)
{
  int t, n;
  float r, denom[S];

  for(t = 0; t < T; t++){
    r = 0;
    for(n = 0; n < N; n++){
      r += z[t*S+n]*z[t*S+n];
    }
    denom[t] = sqrt(r);
  }
  for(t = T; t < S; t++){
    denom[t] = 0;
  }

  float *zd, *denomd, *sid;
  int size = S*S*sizeof(float);
  cudaError_t stat;

  stat = cudaMalloc((void**)&zd, size);
  if (stat != cudaSuccess){
    puts("1");
  }
  stat = cudaMalloc((void**)&denomd, S*sizeof(float));
  if (stat != cudaSuccess){
    puts("2");
  }
  stat =  cudaMalloc((void**)&sid, size);
  if (stat != cudaSuccess){
    puts("3");
  }
  stat = cudaMemcpy(zd, z, size, cudaMemcpyHostToDevice);
  if (stat != cudaSuccess){
    puts("4");
  }
  stat = cudaMemcpy(denomd, denom, S*sizeof(float), cudaMemcpyHostToDevice);
  if (stat != cudaSuccess){
    puts("5");
  }

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(S/BLOCK_SIZE, S/BLOCK_SIZE);
  Kernel<<<dimGrid, dimBlock>>>(zd, denomd, sid);

  stat = cudaMemcpy(si, sid, size, cudaMemcpyDeviceToHost);
  if (stat != cudaSuccess){
    puts("6");
  }

  cudaFree(zd);
  cudaFree(denomd);
  cudaFree(sid);
}
void output(char *filename, float *si)
{
  FILE *file;
  int t1, t2;

  file = fopen(filename, "w");
  for(t1 = 0; t1 < T; t1++){
    for(t2 = 0; t2 < T; t2++){
      fprintf(file, "%d %d %f\n", t1, t2, si[t1*S+t2]);
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

  z = (float *)malloc(S*S*sizeof(float));
  si = (float *)malloc(S*S*sizeof(float));

  initialize(argv[1], z);
  similarity_index(z, si);
  output(argv[2], si);

  free(z);
  free(si);

  return 0;
}

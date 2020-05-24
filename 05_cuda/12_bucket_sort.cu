#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __managed__ int sum;

__global__ void thread(int *a) { 
  a[threadIdx.x] = 0; 
}

__global__ void reduction(int *bucket,int *key) {
  int i = threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void sort(int num,int *key,int sum) {
  int thread = threadIdx.x;
  key[sum+thread] = num;
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  thread<<<1,range>>>(bucket);
  reduction<<<1,n>>>(bucket,key);
  sum=0;
  for(int i=0;i<range;i++){
    int threadnum = bucket[i];
    sort<<<1,threadnum>>>(i,key,sum);
    sum += threadnum;
  }
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
<<<<<<< HEAD
  for(int i=0; i<N; i++) {
    //calculate rx,ry,r^2
    __m256 xnvec = _mm256_load_ps(x);
    __m256 ynvec = _mm256_load_ps(y);
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(xivec,xnvec);
    __m256 ryvec = _mm256_sub_ps(yivec,ynvec);
    __m256 rx2vec = _mm256_mul_ps(rxvec,rxvec);
    __m256 ry2vec = _mm256_mul_ps(ryvec,ryvec);
    __m256 r2vec = _mm256_add_ps(rx2vec,ry2vec);
    
    //calculate 1/r^3
    //use mask to avoid nan
    __m256 same = _mm256_set1_ps(1e-15);
    __m256 mask = _mm256_cmp_ps(r2vec,same,_CMP_GT_OQ);
    __m256 vec0 = _mm256_set1_ps(0);
    __m256 urvec = _mm256_rsqrt_ps(r2vec);
    urvec = _mm256_blendv_ps(vec0,urvec,mask);
    __m256 ur2vec = _mm256_mul_ps(urvec,urvec);
    __m256 ur3vec = _mm256_mul_ps(urvec,ur2vec);

    //calculate rx*m,ry*m 
    __m256 mvec = _mm256_load_ps(m);
    __m256 tfxvec = _mm256_mul_ps(rxvec,mvec);
    __m256 tfyvec = _mm256_mul_ps(ryvec,mvec);
    
    //calculate fx,fy
    __m256 fxvec = _mm256_mul_ps(tfxvec,ur3vec);
    __m256 fyvec = _mm256_mul_ps(tfyvec,ur3vec);
   
    //add all
    float fxi[N],fyi[N];
    _mm256_store_ps(fxi, fxvec);
    _mm256_store_ps(fyi, fyvec);
    for (int j=0; j<N; j++){
      fx[i] -= fxi[j];
      fy[i] -= fyi[j];
=======
  __m256 zero = _mm256_setzero_ps();
  for(int i=0; i<N; i+=8) {
    __m256 xi = _mm256_load_ps(x+i);
    __m256 yi = _mm256_load_ps(y+i);
    __m256 fxi = zero;
    __m256 fyi = zero;
    for(int j=0; j<N; j++) {
      __m256 dx = _mm256_set1_ps(x[j]);
      __m256 dy = _mm256_set1_ps(y[j]);
      __m256 mj = _mm256_set1_ps(m[j]);
      __m256 r2 = zero;
      dx = _mm256_sub_ps(xi, dx);
      dy = _mm256_sub_ps(yi, dy);
      r2 = _mm256_fmadd_ps(dx, dx, r2);
      r2 = _mm256_fmadd_ps(dy, dy, r2);
      __m256 mask = _mm256_cmp_ps(r2, zero, _CMP_GT_OQ);
      __m256 invR = _mm256_rsqrt_ps(r2);
      invR = _mm256_blendv_ps(zero, invR, mask);
      mj = _mm256_mul_ps(mj, invR);
      invR = _mm256_mul_ps(invR, invR);
      mj = _mm256_mul_ps(mj, invR);
      fxi = _mm256_fmadd_ps(dx, mj, fxi);
      fyi = _mm256_fmadd_ps(dy, mj, fyi);
>>>>>>> f743798ff25f63cf544466b630c34b35525ca76f
    }
    _mm256_store_ps(fx+i, fxi);
    _mm256_store_ps(fy+i, fyi);
  }
  for(int i=0; i<N; i++)
    printf("%d %g %g\n",i,fx[i],fy[i]);
}

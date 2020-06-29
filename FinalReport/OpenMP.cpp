#include <math.h>
#include <stdio.h>


#define NX 41 //41
#define NY 41 //41
#define NT 500
#define NIT 50
#define C 1

void build_up_b( double b[NY][NX],   double rho,   double dt,   double u[NY][NX],   double v[NY][NX],   double dx,   double dy);
void pressure_poisson(  double p[NY][NX],   double dx,   double dy,   double b[NY][NX]);
void cavity_flow(int nt,   double u[NY][NX],  double v[NY][NX],  double dt,  double dx,  double dy,  double p[NY][NX],  double rho,  double nu);

int main(void){
  double nx = NX,ny = NY,nt = NT,nit = NIT,c = C;
  double dx = 2 / (nx - 1);
  double dy = 2 / (ny - 1);
  //printf("%Lf",dy);
  double x[NX]={},y[NY]={};
  for(int i=0;i<nx;i++){
    x[i] = 2/nx*i;
  }
  for(int i=0;i<ny;i++){
    y[i] = 2/ny*i;
  }
  //X, Y = numpy.meshgrid(x, y);

  double rho = 1,nu = 0.1,dt = 0.001;

  double u[NY][NX] = {};
  double v[NY][NX] = {};
  double p[NY][NX] = {};
  double b[NY][NX] = {};

  nt = 100;

  cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu); //to get u,v,p

  /*
  for(int i=0;i<NY;i++){
    for(int j=0;j<NX;j++){
      printf("%f ",u[i][j]);
      printf("%f ",v[i][j]);
      printf("%f ",p[i][j]);
    }
    printf("\n");
  }
  */
  

  return 0;
}

void build_up_b( double b[NY][NX],  double rho,  double dt,  double u[NY][NX],  double v[NY][NX],  double dx,  double dy){
#pragma omp parallel for
  for(int i=1;i<NY-1;i++){
#pragma omp parallel for
    for(int j=1;j<NX-1;j++){
      b[i][j] = (rho * (1 / dt *
                      ((u[i][j+1] - u[i][j-1]) /
                       (2 * dx) + (v[i+1][j] - v[i-1][j]) / (2 * dy)) -
                      pow(((u[i][j+1] - u[i][j-1]) / (2 * dx)),2) -    //pow(x, 2.0);
                        2 * ((u[i+1][j] - u[i-1][j]) / (2 * dy) *
                        (v[i][j+1] - v[i][j-1]) / (2 * dx))-
                        pow(((v[i+1][j] - v[i-1][j]) / (2 * dy)),2))); //pow(x, 2.0);
    }
  }

}

void pressure_poisson( double p[NY][NX], double dx, double dy,  double b[NY][NX]){
    int i,j,q;
    double pn[NY][NX]={};
#pragma omp parallel for
    for(i=0;i<NY;i++){
#pragma omp parallel for
      for(j=0;j<NX;j++){
        pn[i][j] = p[i][j];
      }
    }
#pragma omp parallel for
    for(q=0;q<NIT;q++){
      //pn = p.copy()
#pragma omp parallel for
      for(i=0;i<NY;i++){
#pragma omp parallel for
        for(j=0;j<NX;j++){
          pn[i][j] = p[i][j];
        }
      }
#pragma omp parallel for
      for(i=1;i<NY-1;i++){
#pragma omp parallel for
        for(j=1;j<NX-1;j++){
          p[i][j] = (((pn[i][j+1] + pn[i][j-1]) * dy * dy +
                       (pn[i+1][j] + pn[i-1][j]) * dx * dx) /
                       (2 * (dx*dx + dy*dy)) -
                        dx*dx * dy*dy / (2 * (dx*dx + dy*dy)) *
                        b[i][j]);


        }
      }
#pragma omp parallel for
      for(i=0;i<NY;i++){
        p[i][NX-1] = p[i][NX-2]; // dp/dx = 0 at x = 2
      }
#pragma omp parallel for
      for(i=0;i<NX;i++){
        p[0][i] = p[1][i];  // dp/dy = 0 at y = 0
      }
#pragma omp parallel for
      for(i=0;i<NY;i++){
        p[i][0] = p[i][1];  // dp/dx = 0 at x = 0
      }
#pragma omp parallel for
      for(i=0;i<NX;i++){
        p[NY-1][i] = 0;
      }
    }
}

void cavity_flow(int nt,  double u[NY][NX],  double v[NY][NX], double dt, double dx, double dy, double p[NY][NX], double rho, double nu){
    int i,j,n;
    double un[NY][NX]={},vn[NY][NX]={},b[NY][NX]={};
#pragma omp parallel for
    for(n=0;n<nt;n++){
#pragma omp parallel for
      for(i=0;i<NY;i++){
#pragma omp parallel for
        for(j=0;j<NX;j++){
          un[i][j] = u[i][j];
          vn[i][j] = v[i][j];
          //printf("%f",un[i][j]);
        }
      }

      build_up_b(b, rho, dt, u, v, dx, dy); //make_b
#pragma omp barrier
      pressure_poisson(p, dx, dy, b); //make_p
#pragma omp barrier

#pragma omp parallel for
      for(i=1;i<NY-1;i++){
#pragma omp parallel for
        for(j=1;j<NX-1;j++){
          u[i][j] = (un[i][j]-
                     un[i][j] * dt / dx *
                    (un[i][j] - un[i][j-1]) -
                     vn[i][j] * dt / dy *
                    (un[i][j] - un[i-1][j]) -
                     dt / (2 * rho * dx) * (p[i][j+1] - p[i][j-1]) +
                     nu * (dt / (dx*dx) *
                    (un[i][j+1] - 2 * un[i][j] + un[i][j-1]) +
                     dt / (dy*dy) *
                    (un[i+1][j] - 2 * un[i][j] + un[i-1][j])));

          v[i][j] = (vn[i][j] -
                    un[i][j] * dt / dx *
                    (vn[i][j] - vn[i][j-1]) -
                    vn[i][j] * dt / dy *
                   (vn[i][j] - vn[i-1][j]) -
                    dt / (2 * rho * dy) * (p[i+1][j] - p[i-1][j]) +
                    nu * (dt / (dx*dx) *
                   (vn[i][j+1] - 2 * vn[i][j] + vn[i][j-1]) +
                    dt / (dy*dy) *
                   (vn[i+1][j] - 2 * vn[i][j] + vn[i-1][j])));
        }
      }
#pragma omp parallel for
      for(i=0;i<NY;i++){
        u[i][0]  = 0;
        u[i][NX-1] = 0;
        v[i][0]  = 0;
        v[i][NX-1] = 0;
      }
#pragma omp parallel for
      for(j=0;j<NX;j++){
        u[0][j]  = 0;
        u[NY-1][j] = 1;    // set velocity on cavity lid equal to 1
        v[0][j] = 0;
        v[NY-1][j] = 0;
      }
    }
}

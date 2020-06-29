#include <math.h>
#include <stdio.h>
#include <mpi.h>

#define NX 41 //41
#define NY 41 //41
#define NT 500
#define NIT 50
#define C 1

//void build_up_b( double b[NY][NX],   double rho,   double dt,   double u[NY][NX],   double v[NY][NX],   double dx,   double dy,int begin,int end);
//void pressure_poisson(  double p[NY][NX],   double dx,   double dy,   double b[NY][NX],int begin,int end);
//void cavity_flow(int nt,   double u[NY][NX],  double v[NY][NX],  double dt,  double dx,  double dy,  double p[NY][NX],  double rho,  double nu,int begin,int end);

int main(int argc, char** argv){
  /////
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /////
  double nx = NX,ny = NY,nt = NT,nit = NIT,c = C;

  /////
  int begin = rank * (NY / size);
  int end = (rank + 1) * (NY / size);
  double r_un[end-begin][NX];
  double r_vn[end-begin][NX];
  double r_b[end-begin][NX];
  double r_p[end-begin][NX];
  double r_u[end-begin][NX];
  double r_v[end-begin][NX];
  for(int i=0;i<end-begin;i++){
    for(int j=0;j<NX;j++){
      r_un[i][j] = 0;
      r_vn[i][j] = 0;
      r_b[i][j] = 0;
      r_p[i][j] = 0;
      r_u[i][j] = 0;
      r_v[i][j] = 0;
    }
  }
  /////

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

  //cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu,begin,end); //to get u,v,p
  ///////////////////////////////////////
  ///////////////////////////////////////
  ///////////////////////////////////////
  int i,j,n;
  double un[NY][NX]={},vn[NY][NX]={};
  for(n=0;n<nt;n++){
    for(i=begin;i<end;i++){
      for(j=0;j<NX;j++){
        r_un[i][j] = u[i][j];
        r_vn[i][j] = v[i][j];
        //printf("%f",un[i][j]);
      }
    }
    MPI_Allgather(&r_un[begin], (end-begin)*NX, MPI_DOUBLE, un, NY*NX, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&r_vn[begin], (end-begin)*NX, MPI_DOUBLE, vn, NY*NX, MPI_DOUBLE, MPI_COMM_WORLD);


    //build_up_b(b, rho, dt, u, v, dx, dy,begin,end); //make_b
    for(i=begin;i<end;i++){
      if(i!=0&&i!=NY-1){
        for(j=1;j<NX-1;j++){
          r_b[i][j] = (rho * (1 / dt *
                          ((u[i][j+1] - u[i][j-1]) /
                           (2 * dx) + (v[i+1][j] - v[i-1][j]) / (2 * dy)) -
                          pow(((u[i][j+1] - u[i][j-1]) / (2 * dx)),2) -    //pow(x, 2.0);
                            2 * ((u[i+1][j] - u[i-1][j]) / (2 * dy) *
                                 (v[i][j+1] - v[i][j-1]) / (2 * dx))-
                                pow(((v[i+1][j] - v[i-1][j]) / (2 * dy)),2))); //pow(x, 2.0);
        }
      }
    }
    MPI_Allgather(&r_b[begin], (end-begin)*NX, MPI_DOUBLE, b, NY*NX, MPI_DOUBLE, MPI_COMM_WORLD);

    //pressure_poisson(p, dx, dy, b,begin,end); //make_p
    int q;
    double pn[NY][NX]={};
    for(i=0;i<NY;i++){
      for(j=0;j<NX;j++){
        pn[i][j] = p[i][j];
      }
    }

    for(q=0;q<NIT;q++){
      for(i=0;i<NY;i++){
        for(j=0;j<NX;j++){
          pn[i][j] = p[i][j];
        }
      }
      for(i=begin;i<end;i++){
        if(i!=0&&i!=NY-1){
          for(j=1;j<NX-1;j++){
            r_p[i][j] = (((pn[i][j+1] + pn[i][j-1]) * dy * dy + //pow(x, 2.0)
                         (pn[i+1][j] + pn[i-1][j]) * dx * dx) /
                         (2 * (dx*dx + dy*dy)) -
                          dx*dx * dy*dy / (2 * (dx*dx + dy*dy)) *
                          b[i][j]);


          }
        }
      }
      for(i=begin;i<end;i++){
        r_p[i][NX-1] = p[i][NX-2]; // dp/dx = 0 at x = 2
      }
      if (rank==0){
        for(i=0;i<NX;i++){
          r_p[0][i] = p[1][i];  // dp/dy = 0 at y = 0
        }
      }
      for(i=begin;i<end;i++){
        r_p[i][0] = p[i][1];  // dp/dx = 0 at x = 0
      }
      for(i=0;i<NX;i++){
        p[NY-1][i] = 0;
      }
      MPI_Allgather(&r_p[begin], (end-begin)*NX, MPI_DOUBLE, p, NY*NX, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    //MPI_Bcast(p, NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(i=begin;i<end;i++){
      if(i!=0&&i!=NY-1){
        for(j=1;j<NX-1;j++){
          r_u[i][j] = (un[i][j]-
                     un[i][j] * dt / dx *
                    (un[i][j] - un[i][j-1]) -
                     vn[i][j] * dt / dy *
                    (un[i][j] - un[i-1][j]) -
                     dt / (2 * rho * dx) * (p[i][j+1] - p[i][j-1]) +
                     nu * (dt / (dx*dx) *
                    (un[i][j+1] - 2 * un[i][j] + un[i][j-1]) +
                     dt / (dy*dy) *
                    (un[i+1][j] - 2 * un[i][j] + un[i-1][j])));

          r_v[i][j] = (vn[i][j] -
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
    }
    for(i=begin;i<end;i++){
      r_u[i][0]  = 0;
      r_u[i][NX-1] = 0;
      r_v[i][0]  = 0;
      r_v[i][NX-1] = 0;
    }
    for(j=0;j<NX;j++){
      u[0][j]  = 0;
      u[NY-1][j] = 1;    // set velocity on cavity lid equal to 1
      v[0][j] = 0;
      v[NY-1][j] = 0;
    }
    MPI_Allgather(&r_u[begin], (end-begin)*NX, MPI_DOUBLE, u, NY*NX, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&r_v[begin], (end-begin)*NX, MPI_DOUBLE, v, NY*NX, MPI_DOUBLE, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  for(int i=0;i<NY;i++){
    for(int j=0;j<3;j++){
      printf("%f ",u[i][j]);
      printf("%f ",v[i][j]);
      printf("%f ",p[i][j]);
    }
    printf("\n");
  }


  return 0;
}

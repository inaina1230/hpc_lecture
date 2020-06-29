20M30841 

最終課題はFinalReportディレクトリの中にある。
C_base.cppは2D Navier-StokesのパイソンのコードをC++に書き直したものである。
また、OpenMP.cppはOpenMPを用いて並列化したプログラム、
OpenACC.cppはOpenACCを用いて並列化したプログラム、
MPI.cppはMPIを用いて並列化を試みたプログラムである。
C_base.cppとOpenMP.cppとOpenACC.cppは出力を行わないが、u,p,vが求まるためそれを用いて2D Navier-Stokesを
書ける。
一方で、MPIを用いての並列化は出力が明らかに異なり、失敗であった。

Codes for final report are in FinalReport directory.
I rewrited the 2D Navier-Stokes python code in C++ on C_base.cpp.
OpenMP.cpp was written using OpenMP to parallelize ,and
OpenACC.cpp was written using OpenACC.
These programs don't output to standard output,but we can reproduce 2D Navier-Stokes 
to use u,p,v in programs.
I tried to parallelize code using MPI on MPI.cpp, but I couldn't solve this problem.
 
# hpc_lecture

|          | Topic                                | Sample code               |
| -------- | ------------------------------------ | ------------------------- |
| Class 1  | Introduction to parallel programming |                           |
| Class 2  | Shared memory parallelization        | 02_openmp                 |
| Class 3  | Distributed memory parallelization   | 03_mpi                    |
| Class 4  | SIMD parallelization                 | 04_simd                   |
| Class 5  | GPU programming                      | 05_cuda,05_openacc        |
| Class 6  | Parallel programing models           | 06_starpu                 |
| Class 7  | Cache blocking                       | 07_cache_cpu,07_cache_gpu |
| Class 8  | High Performance Python              | 08_python                 |
| Class 9  | I/O libraries                        | 09_io                     |
| Class 10 | Parallel debugger                    | 10_debugger               |
| Class 11 | Parallel profiler                    | 11_profiler               |
| Class 12 | Containers                           |                           |
| Class 13 | Scientific computing                 | 13_pde                    |
| Class 14 | Deep Learning                        | 14_dl                     |

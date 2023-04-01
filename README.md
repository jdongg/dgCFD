# dgCFD

This is an implementation of a modal discontinuous Galerkin finite element method for the two-dimensional compressible Euler equations. Time discretization is done with an explicit third-order Runge Kutta scheme. Slope limiting after each RK stage is done with a moment limiter (see [[1]]; positivity-preserving limiters have been implemented as well).

Implementation exists in MATLAB, C++ (with OpenMP parallelization), and CUDA C++ (supports two-GPU units at present with domain decomposition).

To build the C++ variants, navigate to the respective subdirectory and build the code with 

```
make main
```

The results for C++ variants are written to the directory `./results` and the `.vtk` files are numbered by cycle. They are best viewed in Paraview.
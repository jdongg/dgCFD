# dgCFD

Authors: [Justin Dong](https://jdongg.github.io), [Anna Lischke](https://www.linkedin.com/in/anna-lischke/)

This is an implementation of a modal discontinuous Galerkin finite element method for the two-dimensional compressible Euler equationson quadrilateral meshes. Time discretization is done with an explicit third-order Runge Kutta scheme. Slope limiting after each RK stage is done with a moment limiter (see [1]; positivity-preserving limiters have been implemented as well). In the future, it may be good to consider more recent results on limiters such as WENO-based limiters.

Implementation exists in MATLAB, C++ (with OpenMP parallelization), and CUDA C++ (supports two-GPU units at present with domain decomposition).

To build the C++ variants, navigate to the respective subdirectory and build the code with 

```
make main
```

The results for C++ variants are written to the directory `./results` and the `.vtk` files are numbered by cycle. They are best viewed in Paraview.

[1] Krivodonova, L. (2007). Limiters for high-order discontinuous Galerkin methods. Journal of Computational Physics, 226(1), 879-896.

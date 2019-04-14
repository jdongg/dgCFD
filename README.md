# dgCFD

This is an implementation of a modal discontinuous Galerkin finite element method for the two-dimensional compressible Euler equations. To compile the C++ version, navigate to the directory `./Euler\ RT\ C++` and build the code:

```
make main
```

Running the executable requires OpenMP. If you do not have OpenMP installed (on Mac OS X, the `g++-8` compiler contains OpenMP), you can remove any references to OpenMP. They are located in the following:

1. In the `Makefile`: remove the compiler flag `-fopenmp`
2. In `./inc/DGsolve.hpp`: remove the `omp.h` header include and remove all OpenMP directives (`#pragma omp parallel for`)

The results are written to the directory `./results` and the `.vtk` files are numbered by cycle. You can batch load the entire group of files into Paraview for easy viewing of the entire simulation - just click "apply" on the left-hand panel and then press the playback button at the top of the window.

Below are tasks that still need to be done:

1. Write a basic CUDA implementation.
2. Write a domain decomposition variant of the program using CUDA and MPI.
3. Optimize the serial `C++` code: this should probably be done first in order to make sure any gains from parallel implementation are optimal. In particular, the computation of the volume and numerical fluxes as well as the minmod functions in the slope limiter are not implemented in an efficient way right now.
4. The minmod limiter is only implemented for linear and quadratic basis functions right now. We should try to at least implement it for cubic basis functions as well since the quadratic basis gave significantly better performance on a much coarser mesh.
5. I'm not convinced the Lax-Friedrichs numerical flux has been implemented completely correctly. In particular, we have something like
```
fh = 0.5*(fLeft + fRight - alpha*(uLeft-uRight)
```

where `alpha` is the magnitude of the maximum eigenvalue of the Jacobian of the flux function. I computed the eigenvalues of the Jacobian at the average of the left and right states on each edge and took the maximum value, but this leads to a fairly restrictive CFL condition that is worse than just doing something stupid like `alpha=1.0`. Maybe we can ask Shu about this if we're stuck. I think it is worth doing a read through of existing literature to see how people compute the `alpha` parameter in practice.
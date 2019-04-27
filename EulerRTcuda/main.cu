#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "meshing.hpp"
#include "utilities.hpp"
#include "DGsolve.hpp"
#include "parameters.hpp"

/* This is the main driver for solving the two-dimensional compressible
 * Euler equations using a modal discontinuous Galerkin method. We primarily
 * consider the Riemann problem on the unit square in which each quadrant contains
 * piecewise constant initial data (see, e.g. Xiu and Osher), and also the
 * Rayleigh-Taylor instability problem on the domain [0,1/4]x[0,1].
 *
 * The simulation parameters are set exclusively in parameters.cpp. Results are
 * written to the directory ./results.
 */
int main() {
	// mesh parameters
	int Nx, Ny;
	double x0, xN, y0, yN;
	meshParameters(x0, xN, y0, yN, Nx, Ny);

	// DG parameters
	int pdeg;
	double T, dt;
	dgParameters(pdeg, T, dt);

	// run the solver and dump the results to vtk
	DGsolver mySolver(x0, xN, y0, yN, Nx, Ny, pdeg, T, dt);
	mySolver.DGsolve();

	return 0;
}
#ifndef DGSOLVE_HPP
#define DGSOLVE_HPP

#include "meshing.hpp"
#include "utilities.hpp"
#include "parameters.hpp"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda.h>

__global__
void computeRHScuda(int Nx, int Ny, int nElems, int nOrder2D, int nOrder1D, int nLoc, int *mapB, double *Q1, double *Q2, 
					double *Q3, double *Q4, double *rhsQ1, double *rhsQ2, double *rhsQ3, double *rhsQ4,
				    double *phiVol, double *dphixVol, double *dphiyVol, double *w2d, double *w1d,
				    double *phiEdgeLeft, double *phiEdgeRight, double *phiEdgeBottom, double *phiEdgeTop);

__global__
void updateCUDA1(int nElems, int nLoc, double dt, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40, double *rhsQ1, double *rhsQ2,
	             double *rhsQ3, double *rhsQ4);

__global__
void updateCUDA2(int nElems, int nLoc, double dt, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40, double *rhsQ1, double *rhsQ2,
	             double *rhsQ3, double *rhsQ4);

__global__
void updateCUDA3(int nElems, int nLoc, double dt, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40, double *rhsQ1, double *rhsQ2,
	             double *rhsQ3, double *rhsQ4);

__global__
void updateCUDA4(int nElems, int nLoc, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40);

__global__
void momentLimiterCUDA(int nElems, int nLoc, int pdeg, int *mapB, double *Q);

/* DGsolver is the overarching class containing the routines to compute
 * the discontinuous Galerkin finite element solution. The default constructor
 * takes the following arguments:
 *
 * \param [in] xleft:  left boundary of domain
 * \param [in] xright: right boundary of domain
 * \param [in] yleft:  bottom boundary of domain
 * \param [in] yright: top boundary of domain
 * \param [in] dimx:   number of mesh elements in x direction
 * \param [in] dimy:   number of mesh elements in y direction
 *
 */
class DGsolver {
  private:
  	int pdeg, nLoc, nNodes, nElems, nOrder1D, nOrder2D, Nx, Ny;
  	double T, dt, dx, dy;
  	double *w2d, *x2d, *w1d, *x1d, *phiVol, *dphixVol, *dphiyVol;
  	// double phiVol[9*16];
  	// double dphixVol[9*16];
  	// double dphiyVol[9*16];
  	double *phiEdgeLeft, *phiEdgeRight, *phiEdgeBottom, *phiEdgeTop;
  	double *Q1, *Q2, *Q3, *Q4, *Q10, *Q20, *Q30, *Q40;
  	double *rhsQ1, *rhsQ2, *rhsQ3, *rhsQ4;

    mesh myMesh;

  public:
  	DGsolver(double xleft, double xright, double yleft, double yright, int dimx, int dimy, int p, double Tf, double timeStep) : 
  			 myMesh(xleft, xright, yleft, yright, dimx, dimy) {
  	  pdeg = p; // degree of basis
  	  T = Tf;	// stopping time
  	  dt = timeStep; // time step

  	  dx = (xright-xleft)/dimx;
  	  dy = (yright-yleft)/dimy;
  	  Nx = dimx;
  	  Ny = dimy;

  	  // construct the mesh given by input parameter dim
  	  myMesh.quadConnectivity2D();

  	  nLoc = (pdeg+1)*(pdeg+1); // degrees of freedom per element
  	  nNodes = myMesh.getNumVerts();
      nElems = myMesh.getNumCells();

      nOrder1D = pdeg+1; // number of quadrature nodes for surface integrals
      nOrder2D = (pdeg+2)*(pdeg+2); // number of quad nodes for volume integrals

      // initialize quadrature points
	  // w2d = new double[nOrder2D];
	  // x2d = new double[nOrder2D*2];
	  // w1d = new double[nOrder1D];
	  // x1d = new double[nOrder1D];
	  cudaMallocManaged(&w2d, nOrder2D*sizeof(double));
	  cudaMallocManaged(&x2d, nOrder2D*2*sizeof(double));
	  cudaMallocManaged(&w1d, nOrder1D*sizeof(double));
	  cudaMallocManaged(&x1d, nOrder1D*sizeof(double));

	  DGsolver::quadRule2D(sqrt(nOrder2D));
	  DGsolver::quadRule1D(nOrder1D);

	  // initialize basis functions in interior
	  // phiVol = new double[nLoc*nOrder2D];
	  // dphixVol = new double[nLoc*nOrder2D];
	  // dphiyVol = new double[nLoc*nOrder2D];
	  cudaMallocManaged(&phiVol, nLoc*nOrder2D*sizeof(double));
	  cudaMallocManaged(&dphixVol, nLoc*nOrder2D*sizeof(double));
	  cudaMallocManaged(&dphiyVol, nLoc*nOrder2D*sizeof(double));
	  DGsolver::basisFunctions(pdeg, nOrder2D, phiVol, x2d);
	  DGsolver::basisFunctionsGrad(pdeg, nOrder2D, dphixVol, dphiyVol, x2d);

	  // initialize basis functions on edges
	  // phiEdgeLeft = new double[nLoc*nOrder1D];
	  // phiEdgeRight = new double[nLoc*nOrder1D];
	  // phiEdgeBottom = new double[nLoc*nOrder1D];
	  // phiEdgeTop = new double[nLoc*nOrder1D];
	  cudaMallocManaged(&phiEdgeLeft, nLoc*nOrder1D*sizeof(double));
	  cudaMallocManaged(&phiEdgeRight, nLoc*nOrder1D*sizeof(double));
	  cudaMallocManaged(&phiEdgeBottom, nLoc*nOrder1D*sizeof(double));
	  cudaMallocManaged(&phiEdgeTop, nLoc*nOrder1D*sizeof(double));
	  DGsolver::basisFunctionsEdge(pdeg);

	  // initialize solution arrays and right-hand side arrays
	  // Q1 = new double[nElems*nLoc];
	  // Q2 = new double[nElems*nLoc];
	  // Q3 = new double[nElems*nLoc];
	  // Q4 = new double[nElems*nLoc];

	  // Q10 = new double[nElems*nLoc]();
	  // Q20 = new double[nElems*nLoc]();
	  // Q30 = new double[nElems*nLoc]();
	  // Q40 = new double[nElems*nLoc]();

	  // rhsQ1 = new double[nElems*nLoc];
	  // rhsQ2 = new double[nElems*nLoc];
	  // rhsQ3 = new double[nElems*nLoc];
	  // rhsQ4 = new double[nElems*nLoc];

	  cudaMallocManaged(&Q1, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&Q2, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&Q3, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&Q4, nElems*nLoc*sizeof(double));

	  cudaMallocManaged(&Q10, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&Q20, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&Q30, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&Q40, nElems*nLoc*sizeof(double));

	  cudaMallocManaged(&rhsQ1, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&rhsQ2, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&rhsQ3, nElems*nLoc*sizeof(double));
	  cudaMallocManaged(&rhsQ4, nElems*nLoc*sizeof(double));

	}

	// default destructor
	~DGsolver() {
	  // delete[] w2d;
	  // delete[] x2d;
	  // delete[] w1d;
	  // delete[] x1d;
	  // delete[] phiVol;
	  // delete[] dphixVol;
	  // delete[] dphiyVol;
	  // delete[] phiEdgeLeft;
	  // delete[] phiEdgeRight;
	  // delete[] phiEdgeBottom;
	  // delete[] phiEdgeTop;
	  // delete[] Q1;
	  // delete[] Q2;
	  // delete[] Q3;
	  // delete[] Q4;
	  // delete[] Q10;
	  // delete[] Q20;
	  // delete[] Q30;
	  // delete[] Q40;
	  // delete[] rhsQ1;
	  // delete[] rhsQ2;
	  // delete[] rhsQ3;
	  // delete[] rhsQ4;

	  cudaFree(w2d);
	  cudaFree(x2d);
	  cudaFree(w1d);
	  cudaFree(x1d);
	  cudaFree(phiVol);
	  cudaFree(dphixVol);
	  cudaFree(dphiyVol);
	  cudaFree(phiEdgeLeft);
	  cudaFree(phiEdgeRight);
	  cudaFree(phiEdgeBottom);
	  cudaFree(phiEdgeTop);
	  cudaFree(Q1);
	  cudaFree(Q2);
	  cudaFree(Q3);
	  cudaFree(Q4);
	  cudaFree(Q10);
	  cudaFree(Q20);
	  cudaFree(Q30);
	  cudaFree(Q40);
	  cudaFree(rhsQ1);
	  cudaFree(rhsQ2);
	  cudaFree(rhsQ3);
	  cudaFree(rhsQ4);
	}

	// member functions of DGsolver
	void DGsolve();
	void L2projection();
	void timeStepper();
	int sgn(double val);
	double myMin(int N, double *array);
	double myMax(int N, double *array);
	void positivityLimiter(double m, double M, double *Q);
	void basisFunctions(int p, int N, double *phi, double *xy);
	void basisFunctionsGrad(int p, int N, double *phidx, double *phidy, double *xy);
	void basisFunctionsEdge(int p);
	void quadRule1D(int Norder);
	void quadRule2D(int Norder);
	void runCUDAcomputeRHS();
	void runCUDAupdate1();
	void runCUDAupdate2();
	void runCUDAupdate3();
	void runCUDAupdate4();
	void runCUDAmomentLimiter(double *Q);

	friend class mesh;
};

/* DGsolve is the main routine that solves the 2D Euler equations.
 * A call to DGsolve projects the initial condition onto the mesh
 * specified in the constructor with basis functions of degree pdeg
 * on each element. DGsolve then calls the time stepping routine
 * to compute a numerical solution at time T.
 */
inline void DGsolver::DGsolve() {
	// compute L2 projection of initial conditions
	DGsolver::L2projection();

	// time stepping; this is the part that should be parallelized
	DGsolver::timeStepper();	
}


/* timeStepper updates the solution using a third-order TVD Runge-Kutta
 * discretization until time T is reached. A slope limiter is applied
 * after each RK stage to enforce total variation boundedness of the
 * numerical solutions.
 */
inline void DGsolver::timeStepper() {
	double t = 0.0;

	// zero stage: copy initial data to Q
	for (int i = 0; i < nElems*nLoc; ++i)
	{
		Q1[i] = Q10[i];
		Q2[i] = Q20[i];
		Q3[i] = Q30[i];
		Q4[i] = Q40[i];
	}


	int iter = 0;
	int plotiter = 0;
	while (t < T) {
		// first RK stage
		DGsolver::runCUDAcomputeRHS();
		DGsolver::runCUDAupdate1();

		// apply moment limiter component-wise
		DGsolver::runCUDAmomentLimiter(Q1);
		DGsolver::runCUDAmomentLimiter(Q2);
		DGsolver::runCUDAmomentLimiter(Q3);
		DGsolver::runCUDAmomentLimiter(Q4);

		// // apply positivity limiter to density
		// DGsolver::positivityLimiter(0.0, 2.0, Q1);

		// second RK stage
		DGsolver::runCUDAcomputeRHS();
		DGsolver::runCUDAupdate2();

		// apply moment limiter
		DGsolver::runCUDAmomentLimiter(Q1);
		DGsolver::runCUDAmomentLimiter(Q2);
		DGsolver::runCUDAmomentLimiter(Q3);
		DGsolver::runCUDAmomentLimiter(Q4);

		// // apply positivity limiter to density
		// DGsolver::positivityLimiter(0.0, 2.0, Q1);

		// third RK stage
		DGsolver::runCUDAcomputeRHS();
		DGsolver::runCUDAupdate3();

		// apply moment limiter component-wise
		DGsolver::runCUDAmomentLimiter(Q10);
		DGsolver::runCUDAmomentLimiter(Q20);
		DGsolver::runCUDAmomentLimiter(Q30);
		DGsolver::runCUDAmomentLimiter(Q40);

		// // apply positivity limiter to density
		// DGsolver::positivityLimiter(0.0, 2.0, Q10);

		// zero stage: copy initial data to Q
		DGsolver::runCUDAupdate4();

		// dump density solution to .vtk output every 20 time steps
		if (iter%240 == 0)
		{
			cudaDeviceSynchronize();
			double *uCell = new double[nOrder2D];
			double *sol = new double[nElems]();
			for (int k = 0; k < nElems; ++k)
			{
				// compute the solution at the 2D quadrature nodes in each element
				for (int i = 0; i < nOrder2D; ++i)
				{
					uCell[i] = 0.0;
					for (int j = 0; j < nLoc; ++j)
					{
						uCell[i] += Q10[k*nLoc+j]*phiVol[j*nOrder2D+i];
					}			
				}

				// take the average of the solution in the element and assign it
				// to the field value in the .vtk file
				for (int i = 0; i < nOrder2D; ++i)
				{
					sol[k] += uCell[i]/nOrder2D;
				}
			}

			delete[] uCell;

			writeVTK<double>(nElems, nNodes, myMesh.getEToV(), myMesh.getVxy(), sol, t, plotiter);
			plotiter++;
		}

		printf("Time: %f\n", t);
		t += dt;
		iter++;
	}
}


/* myMin is a naive implementation of the minimum function.
 * 
 * \param [in] N:     length of array
 * \param [in] array: buffer of values to take minimum over
 * \param [out]:	  minimum value of array
 */
inline double DGsolver::myMin(int N, double *array) {
	// set the initial minimum artificially high
	double min = 10000.0;
 
 	for (int i = 0; i < N; ++i) {
 		if (array[i] < min) {
 			min = array[i];
 		}
 	}

 	return min;
}


/* myMax is a naive implementation of the maximum function.
 * 
 * \param [in] N:     length of array
 * \param [in] array: buffer of values to take minimum over
 * \param [out]:	  maximum value of array
 */
inline double DGsolver::myMax(int N, double *array) {
	// set the initial minimum artificially high
	double max = -10000.0;
 
 	for (int i = 0; i < N; ++i) {
 		if (array[i] > max) {
 			max = array[i];
 		}
 	}

 	return max;
}


/* positivityLimiter implements the maximum principle preserving limiter, which
 * enforces that the numerical solution always lie between prescribed values m and
 * M.

 */
inline void DGsolver::positivityLimiter(double m, double M, double *Q) {

	double *phiVert = new double[nLoc*4];
	double xyVert[8] = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
	DGsolver::basisFunctions(pdeg, 4, phiVert, xyVert);

	for (int k = 0; k < nElems; ++k)
	{
		// grab coordinates of vertices on element k
		double xa, xb, ya, yb, vol;

		xa = myMesh.Vxy[myMesh.EToV[k*6]*2];
		xb = myMesh.Vxy[myMesh.EToV[k*6+1]*2];
		ya = myMesh.Vxy[myMesh.EToV[k*6]*2+1];
		yb = myMesh.Vxy[myMesh.EToV[k*6+2]*2+1];

		// Jacobian of transformation from reference element to
		// element k
		vol = (xb-xa)*(yb-ya)/4.0;

		// grab coefficients of DG solution on element k
		double *c = new double[nLoc];
		double *cmod = new double[nLoc];
		double *uloc = new double[nOrder2D]();

		for (int i = 0; i < nLoc; ++i)
		{
			c[i] = Q[k*nLoc+i];
		}

		// compute the values of the conserved quantity at the 2D quadrature points
		for (int i = 0; i < nOrder2D; ++i)
		{
			for (int j = 0; j < nLoc; ++j)
			{
				uloc[i] += c[j]*phiVol[j*nOrder2D+i];
			}
		}

		// compute cell average
		double avg = 0.0;
		for (int i = 0; i < nOrder2D; ++i)
		{
			avg += vol*w2d[i]*uloc[i]/((xb-xa)*(yb-ya));
		}


		// also compute values of the conserved quantity at the vertices
		double uvert[4] = {0.0, 0.0, 0.0, 0.0};
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < nLoc; ++j)
			{
				uvert[i] += c[j]*phiVert[j*4+i];
			}
		}

		// append values
		double *ulocVert = new double[nOrder2D+4];
		for (int i = 0; i < nOrder2D; ++i)
		{
			ulocVert[i] = uloc[i];
		}

		ulocVert[nOrder2D] = uvert[0];
		ulocVert[nOrder2D+1] = uvert[1];
		ulocVert[nOrder2D+2] = uvert[2];
		ulocVert[nOrder2D+3] = uvert[3];

		// compute max and min values on cell k
		double mk = myMin(nOrder2D+4, ulocVert);
		double Mk = myMax(nOrder2D+4, ulocVert);

		// modify the polynomial according to the MPP limiter. note that all 
		// coefficients are simply scaled by theta while the first coefficient 
		// corresponding to the constant basis function is translated by
		// -2.0*avg*(theta-1). The extra factor of 2 is because the constant 
		// basis function is phi = 1/2.
		double argsTmp[3] = {1.0, fabs((M-avg)/(Mk-avg)), fabs((m-avg)/(mk-avg))};
		double theta = myMin(3, argsTmp);

		for (int i = 0; i < nLoc; ++i)
		{
			cmod[i] = theta*c[i];
		}
		cmod[0] = cmod[0] - 2.0*avg*(theta-1.0);


		for (int i = 0; i < nLoc; ++i)
		{
			Q[k*nLoc+i] = cmod[i];
		}

		delete[] c;
		delete[] cmod;
		delete[] uloc;
		delete[] ulocVert;
	}

	delete[] phiVert;
}

/* L2projection projects the initial data onto the finite element space.
 * In particular, it returns the coefficients of the basis expansion of
 * of the initial condition on each element of the mesh.
 */
inline void DGsolver::L2projection() {
	double xa, xb, ya, yb;
	double xk, yk;
	double q0[4];

	for (int k = 0; k < nElems; ++k)
	{
		xa = myMesh.Vxy[myMesh.EToV[k*6]*2];
		xb = myMesh.Vxy[myMesh.EToV[k*6+1]*2];
		ya = myMesh.Vxy[myMesh.EToV[k*6]*2+1];
		yb = myMesh.Vxy[myMesh.EToV[k*6+2]*2+1];

		for (int j = 0; j < nOrder2D; ++j)
		{
			xk = (xb-xa)/2.0*x2d[j*2] + (xb+xa)/2.0;
			yk = (yb-ya)/2.0*x2d[j*2+1] + (yb+ya)/2.0;

			initialCondition(xk,yk,q0);

			for (int i = 0; i < nLoc; ++i)
			{
				Q10[k*nLoc+i] += w2d[j]*q0[0]*phiVol[i*nOrder2D+j];
				Q20[k*nLoc+i] += w2d[j]*q0[1]*phiVol[i*nOrder2D+j];
				Q30[k*nLoc+i] += w2d[j]*q0[2]*phiVol[i*nOrder2D+j];
				Q40[k*nLoc+i] += w2d[j]*q0[3]*phiVol[i*nOrder2D+j];
			}
			
		}
	}
}

/* basisFunctions evaluates the basis functions on the reference square [-1,1]^2
 * at the points given in xy. The basis functions used are the tensor product 
 * Legendre polynomials.
 *
 * \param [in] p:       degree of the basis
 * \param [in] N:       number of points to evaluate basis functions at
 * \param [in/out] phi: (p+1)^2 x N buffer to hold basis function values
 * \param [in] xy:		Nx2 array of 2D points to evaluate basis at; first column
 *						corresponds to x and second corresponds to y
 */
inline void DGsolver::basisFunctions(int p, int N, double *phi, double *xy) {
	double *phiTmpX = new double[(p+1)*N];
	double *phiTmpY = new double[(p+1)*N];

	// first compute the one-dimensional Legendre polynomials in x and y
	if (p == 0)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpY[i] = 1.0;
		}
	}
	else if (p == 1)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];

			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
		}
	}
	else if (p == 2)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpX[2*N+i] = 0.5*(3.0*xy[i*2]*xy[i*2]-1.0);

			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
			phiTmpY[2*N+i] = 0.5*(3.0*xy[i*2+1]*xy[i*2+1]-1.0);
		}
	}
	else if (p == 3)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpX[2*N+i] = 0.5*(3.0*xy[i*2]*xy[i*2]-1.0);
			phiTmpX[3*N+i] = 0.5*(5.0*xy[i*2]*xy[i*2]*xy[i*2] - 3.0*xy[i*2]);

			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
			phiTmpY[2*N+i] = 0.5*(3.0*xy[i*2+1]*xy[i*2+1]-1.0);
			phiTmpY[3*N+i] = 0.5*(5.0*xy[i*2+1]*xy[i*2+1]*xy[i*2+1] - 3.0*xy[i*2+1]);
		}
	}
	else if (p == 4)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpX[2*N+i] = 0.5*(3.0*xy[i*2]*xy[i*2]-1.0);
			phiTmpX[3*N+i] = 0.5*(5.0*xy[i*2]*xy[i*2]*xy[i*2] - 3.0*xy[i*2]);
			phiTmpX[4*N+i] = (35.0*pow(xy[i*2],4.0) - 30.0*pow(xy[i*2],2.0) + 3.0)/8.0;

			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
			phiTmpY[2*N+i] = 0.5*(3.0*xy[i*2+1]*xy[i*2+1]-1.0);
			phiTmpY[3*N+i] = 0.5*(5.0*xy[i*2+1]*xy[i*2+1]*xy[i*2+1] - 3.0*xy[i*2+1]);
			phiTmpY[4*N+i] = (35.0*pow(xy[i*2+1],4.0) - 30.0*pow(xy[i*2+1],2.0) + 3.0)/8.0;
		}
	}

	// compute normalized tensor products of Legendre polynomials. c is chosen
	// so that ||phi||_{L^{2}} = 1. This property is crucial since the basis is now
	// orthonormal and the mass matrix is a scaled identity matrix. 
	for (int i = 0; i < p+1; ++i)
	{
		for (int j = 0; j < p+1; ++j)
		{
			double c = sqrt((2.0*i+1.0)*(2.0*j+1.0))/2.0;
			for (int k = 0; k < N; ++k)
			{
				int idx = i*(p+1)+j;
				phi[idx*N+k] = c*phiTmpX[i*N+k]*phiTmpY[j*N+k];
			}
		}
	}

	delete[] phiTmpX;
	delete[] phiTmpY;
}


/* basisFunctionsGrad evaluates the first order derivatives of the basis 
 * functions on the reference square [-1,1]^2 at the points given in xy. 
 * The basis functions used are the tensor product Legendre polynomials.
 *
 * \param [in] p:         degree of the basis
 * \param [in] N:         number of points to evaluate basis functions at
 * \param [in/out] phidx: (p+1)^2 x N buffer to hold values of dphi/dx
 * \param [in/out] phidy: (p+1)^2 x N buffer to hold values of dphi/dy
 * \param [in] xy:		  Nx2 array of 2D points to evaluate basis at; first column
 *						  corresponds to x and second corresponds to y
 */
inline void DGsolver::basisFunctionsGrad(int p, int N, double *phidx, double *phidy, double *xy) {
	double *phiTmpX = new double[(p+1)*N];
	double *phiTmpY = new double[(p+1)*N];
	double *dphiTmpX = new double[(p+1)*N];
	double *dphiTmpY = new double[(p+1)*N];

	// first compute dphi/dx and dphi/dy for the one-dimensional
	// basis functions in x and y
	if (p == 0)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpY[i] = 1.0;

			dphiTmpX[i] = 0.0;
			dphiTmpY[i] = 0.0;
		}
	}
	else if (p == 1)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];

			dphiTmpX[i] = 0.0;
			dphiTmpX[N+i] = 1.0;
			dphiTmpY[i] = 0.0;
			dphiTmpY[N+i] = 1.0;
		}
	}
	else if (p == 2)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpX[2*N+i] = 0.5*(3.0*xy[i*2]*xy[i*2]-1.0);
			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
			phiTmpY[2*N+i] = 0.5*(3.0*xy[i*2+1]*xy[i*2+1]-1.0);

			dphiTmpX[i] = 0.0;
			dphiTmpX[N+i] = 1.0;
			dphiTmpX[2*N+i] = 3.0*xy[i*2];
			dphiTmpY[i] = 0.0;
			dphiTmpY[N+i] = 1.0;
			dphiTmpY[2*N+i] = 3.0*xy[i*2+1];
		}
	}
	else if (p == 3)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpX[2*N+i] = 0.5*(3.0*xy[i*2]*xy[i*2]-1.0);
			phiTmpX[3*N+i] = 0.5*(5.0*xy[i*2]*xy[i*2]*xy[i*2] - 3.0*xy[i*2]);
			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
			phiTmpY[2*N+i] = 0.5*(3.0*xy[i*2+1]*xy[i*2+1]-1.0);
			phiTmpY[3*N+i] = 0.5*(5.0*xy[i*2+1]*xy[i*2+1]*xy[i*2+1] - 3.0*xy[i*2+1]);

			dphiTmpX[i] = 0.0;
			dphiTmpX[N+i] = 1.0;
			dphiTmpX[2*N+i] = 3.0*xy[i*2];
			dphiTmpX[3*N+i] = 15.0*pow(xy[i*2],2.0)/2.0 - 1.5*xy[i*2];
			dphiTmpY[i] = 0.0;
			dphiTmpY[N+i] = 1.0;
			dphiTmpY[2*N+i] = 3.0*xy[i*2+1];
			dphiTmpY[3*N+i] = 15.0*pow(xy[i*2+1],2.0)/2.0 - 1.5*xy[i*2+1];
		}
	}
	else if (p == 4)
	{
		for (int i = 0; i < N; ++i)
		{
			phiTmpX[i] = 1.0;
			phiTmpX[N+i] = xy[i*2];
			phiTmpX[2*N+i] = 0.5*(3.0*xy[i*2]*xy[i*2]-1.0);
			phiTmpX[3*N+i] = 0.5*(5.0*xy[i*2]*xy[i*2]*xy[i*2] - 3.0*xy[i*2]);
			phiTmpX[4*N+i] = (35.0*pow(xy[i*2],4.0) - 30.0*pow(xy[i*2],2.0) + 3.0)/8.0;
			phiTmpY[i] = 1.0;
			phiTmpY[N+i] = xy[i*2+1];
			phiTmpY[2*N+i] = 0.5*(3.0*xy[i*2+1]*xy[i*2+1]-1.0);
			phiTmpY[3*N+i] = 0.5*(5.0*xy[i*2+1]*xy[i*2+1]*xy[i*2+1] - 3.0*xy[i*2+1]);
			phiTmpY[4*N+i] = (35.0*pow(xy[i*2+1],4.0) - 30.0*pow(xy[i*2+1],2.0) + 3.0)/8.0;

			dphiTmpX[i] = 0.0;
			dphiTmpX[N+i] = 1.0;
			dphiTmpX[2*N+i] = 3.0*xy[i*2];
			dphiTmpX[3*N+i] = 15.0*pow(xy[i*2],2.0)/2.0 - 1.5*xy[i*2];
			dphiTmpX[4*N+i] = (140.0*pow(xy[i*2],3.0) - 60.0*pow(xy[i*2],2.0))/8.0;
			dphiTmpY[i] = 0.0;
			dphiTmpY[N+i] = 1.0;
			dphiTmpY[2*N+i] = 3.0*xy[i*2+1];
			dphiTmpY[3*N+i] = 15.0*pow(xy[i*2+1],2.0)/2.0 - 1.5*xy[i*2+1];
			dphiTmpY[4*N+i] = (140.0*pow(xy[i*2+1],3.0) - 60.0*pow(xy[i*2+1],2.0))/8.0;
		}
	}

	// compute the normalized tensor product derivatives
	for (int i = 0; i < p+1; ++i)
	{
		for (int j = 0; j < p+1; ++j)
		{
			double c = sqrt((2.0*i+1.0)*(2.0*j+1.0))/2.0;
			for (int k = 0; k < N; ++k)
			{
				int idx = i*(p+1)+j;
				phidx[idx*N+k] = c*dphiTmpX[i*N+k]*phiTmpY[j*N+k]*2.0/dx;
				phidy[idx*N+k] = c*phiTmpX[i*N+k]*dphiTmpY[j*N+k]*2.0/dy;
			}
		}
	}

	delete[] phiTmpX;
	delete[] phiTmpY;
	delete[] dphiTmpX;
	delete[] dphiTmpY;
}


/* basisFunctionsEdge computes the basis function values on each edge of the
 * reference square [-1,1]^2.
 *
 * \param [in] p: degree of basis functions
 */
inline void DGsolver::basisFunctionsEdge(int p) {
	// temporary buffer to store 2D quadrature nodes along each edge
	double *xy1D = new double[nOrder1D*2];

	// left edge
	for (int i = 0; i < nOrder1D; ++i)
	{
		xy1D[i*2] = -1.0;
		xy1D[i*2+1] = x1d[i];
	}
	DGsolver::basisFunctions(p, nOrder1D, phiEdgeLeft, xy1D);

	// bottom edge
	for (int i = 0; i < nOrder1D; ++i)
	{
		xy1D[i*2] = x1d[i];
		xy1D[i*2+1] = -1.0;
	}
	DGsolver::basisFunctions(p, nOrder1D, phiEdgeBottom, xy1D);

	// right edge
	for (int i = 0; i < nOrder1D; ++i)
	{
		xy1D[i*2] = 1.0;
		xy1D[i*2+1] = x1d[i];
	}
	DGsolver::basisFunctions(p, nOrder1D, phiEdgeRight, xy1D);

	// top edge
	for (int i = 0; i < nOrder1D; ++i)
	{
		xy1D[i*2] = x1d[i];
		xy1D[i*2+1] = 1.0;
	}
	DGsolver::basisFunctions(p, nOrder1D, phiEdgeTop, xy1D);

	delete[] xy1D;
}


/* quadRule1D generates Norder Gaussian quadrature points on the interval
 * (-1,1). The rule is interior and does not utilized points on the boundary.
 * The rule is exact for degree 2*Norder-1 polynomials.
 *
 * \param [in] Norder: order of the quadrature rule
 */
inline void DGsolver::quadRule1D(int Norder) {
	if (Norder == 1)
	{
		w1d[0] = 2.0;
		x1d[0] = 0.0;
	}
	else if (Norder == 2)
	{
		w1d[0] = 1.0;
		w1d[1] = 1.0;

		x1d[0] = -0.577350269189;
		x1d[1] = 0.577350269189;
	}
	else if (Norder == 3)
	{
		w1d[0] = 0.555555555555;
        w1d[1] = 0.888888888888;
        w1d[2] = 0.555555555555;

		x1d[0] = -0.774596669241;
        x1d[1] = 0.0;
        x1d[2] = 0.774596669241;
	}
	else if (Norder == 4)
	{
		w1d[0] = 0.347854845137;
        w1d[1] = 0.652145154862;
        w1d[2] = 0.652145154862;
        w1d[3] = 0.347854845137;

        x1d[0] = -0.861136311954;
        x1d[1] = -0.339981043584;
        x1d[2] = 0.339981043584;
        x1d[3] = 0.861136311954;
	}
	else if (Norder == 5)
	{
		w1d[0] = 0.568888888888888;
        w1d[1] = 0.236926885056189;
        w1d[2] = 0.478628670499366;
        w1d[3] = 0.478628670499366;
        w1d[4] = 0.236926885056189;

        x1d[0] = 0.0;
        x1d[1] = -0.906179845938664;
        x1d[2] = -0.538469310105683;
        x1d[3] = 0.538469310105683;
        x1d[4] = 0.906179845938664;
	}
}


/* quadRule2D generates Norder^2 quadrature points on the reference square
 * [-1,1]^2. The quadrature points are the cartesian products of the 1D
 * Gaussian quadrature points with weights given by tensor products of the 
 * 1D weights.
 *
 * \param [in] Norder: order of the underlying 1D quadrature rule. There will
 * 					   be Norder^2 2D points in total.
 */
inline void DGsolver::quadRule2D(int Norder) {
  double *xTmp = new double[Norder];
  double *wTmp = new double[Norder];

  // first generate the 1D points and weights
  if (Norder == 1)
  {
 	wTmp[0] = 2.0;
	xTmp[0] = 0.0;
  }
  else if (Norder == 2)
  {
	wTmp[0] = 1.0;
	wTmp[1] = 1.0;

	xTmp[0] = -0.577350269189;
	xTmp[1] = 0.577350269189;
  } 
  else if (Norder == 3)
  {
	wTmp[0] = 0.555555555555;
    wTmp[1] = 0.888888888888;
    wTmp[2] = 0.555555555555;

	xTmp[0] = -0.774596669241;
    xTmp[1] = 0.0;
    xTmp[2] = 0.774596669241;
  }
  else if (Norder == 4)
  {
	wTmp[0] = 0.347854845137;
    wTmp[1] = 0.652145154862;
    wTmp[2] = 0.652145154862;
    wTmp[3] = 0.347854845137;

    xTmp[0] = -0.861136311954;
    xTmp[1] = -0.339981043584;
    xTmp[2] = 0.339981043584;
    xTmp[3] = 0.861136311954;
  }
  else if (Norder == 5)
  {
	wTmp[0] = 0.568888888888888;
    wTmp[1] = 0.236926885056189;
    wTmp[2] = 0.478628670499366;
    wTmp[3] = 0.478628670499366;
    wTmp[4] = 0.236926885056189;

    xTmp[0] = 0.0;
    xTmp[1] = -0.906179845938664;
    xTmp[2] = -0.538469310105683;
    xTmp[3] = 0.538469310105683;
    xTmp[4] = 0.906179845938664;
  }

  // cartesian products of the 1D points
  for (int i = 0; i < Norder; ++i)
  {
  	for (int j = 0; j < Norder; ++j)
  	{
  		int idx = i*Norder+j;
  		w2d[idx] = wTmp[i]*wTmp[j];
  		x2d[idx*2] = xTmp[i];
  		x2d[idx*2+1] = xTmp[j];
  	}
  }
  delete[] xTmp;
  delete[] wTmp;
}


inline void DGsolver::runCUDAcomputeRHS() {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeRHScuda, 0, nElems); 
	int numBlocks = (nElems + blockSize - 1)/(blockSize);
	// int numBlocks = 1;

	// int blockSize = 32;
	// int numBlocks = 8;

	// cudaDeviceSynchronize();

	// // check for errors
	// cudaError_t error = cudaGetLastError();
	// if (error != cudaSuccess) {
	// 	fprintf(stderr, "ERROR RHS: %d %s \n", blockSize, cudaGetErrorString(error));
	// }

	computeRHScuda<<<numBlocks, blockSize>>>(Nx, Ny, nElems, nOrder2D, nOrder1D, nLoc, myMesh.mapB, Q1, Q2,
										  Q3, Q4, rhsQ1, rhsQ2, rhsQ3, rhsQ4, phiVol, dphixVol, dphiyVol, w2d, w1d, 
										  phiEdgeLeft, phiEdgeRight, phiEdgeBottom, phiEdgeTop);
	
}


inline void DGsolver::runCUDAupdate1() {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA1, 0, nElems*nLoc); 
	int numBlocks = (nElems*nLoc + blockSize - 1)/(blockSize);

	updateCUDA1<<<numBlocks, blockSize>>>(nElems, nLoc, dt, Q1, Q2, Q3, Q4, Q10, Q20, Q30, Q40, rhsQ1, rhsQ2, rhsQ3, rhsQ4);
	// cudaDeviceSynchronize();
}


inline void DGsolver::runCUDAupdate2() {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA2, 0, nElems*nLoc); 
	int numBlocks = (nElems*nLoc + blockSize - 1)/(blockSize);

	updateCUDA2<<<numBlocks, blockSize>>>(nElems, nLoc, dt, Q1, Q2, Q3, Q4, Q10, Q20, Q30, Q40, rhsQ1, rhsQ2, rhsQ3, rhsQ4);
	// cudaDeviceSynchronize();
}


inline void DGsolver::runCUDAupdate3() {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA3, 0, nElems*nLoc); 
	int numBlocks = (nElems*nLoc + blockSize - 1)/(blockSize);

	updateCUDA3<<<numBlocks, blockSize>>>(nElems, nLoc, dt, Q1, Q2, Q3, Q4, Q10, Q20, Q30, Q40, rhsQ1, rhsQ2, rhsQ3, rhsQ4);
	// cudaDeviceSynchronize();
}


inline void DGsolver::runCUDAupdate4() {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA4, 0, nElems*nLoc); 
	int numBlocks = (nElems*nLoc + blockSize - 1)/(blockSize);

	// int numBlocks = 1;
	updateCUDA4<<<numBlocks, blockSize>>>(nElems, nLoc, Q1, Q2, Q3, Q4, Q10, Q20, Q30, Q40);
}


inline void DGsolver::runCUDAmomentLimiter(double *Q) {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, momentLimiterCUDA, 0, nElems); 
	int numBlocks = (nElems + blockSize - 1)/(blockSize);
	// int numBlocks = 1;

	momentLimiterCUDA<<<numBlocks, blockSize>>>(nElems, nLoc, pdeg, myMesh.mapB, Q);

	// cudaDeviceSynchronize();

	// // check for errors
	// cudaError_t error = cudaGetLastError();
	// if (error != cudaSuccess) {
	// 	fprintf(stderr, "ERROR LIMITER: %s \n", cudaGetErrorString(error));
	// }
}

#include "../cuda/DGsolveCUDA.cu"

#endif
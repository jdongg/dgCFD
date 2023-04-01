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

/*__global__
void computeRHScuda(int Nx, int Ny, int nElems, int nOrder2D, int nOrder1D, int nLoc, int *mapB, double *Q1, double *Q2, 
	             double *Q3, double *Q4, double *rhsQ1, double *rhsQ2, double *rhsQ3, double *rhsQ4,
		     double *phiVol, double *dphixVol, double *dphiyVol, double *w2d, double *w1d,
	   	     double *phiEdgeLeft, double *phiEdgeRight, double *phiEdgeBottom, double *phiEdgeTop);
*/
__global__
void computeRHScuda(int subDflg, int Nx, int Ny, int SubnElems, int nOrder2D, int nOrder1D, int nLoc, double *Q1, double *Q2, double *Q3,
                                      double *Q4, double *Q1Halo, double *Q2Halo, double *Q3Halo, double *Q4Halo,
                                      double *rhsQ1, double *rhsQ2, double *rhsQ3, double *rhsQ4, double *phiVol, 
                                      double *dphixVol, double *dphiyVol, double *w2d, double *w1d,
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
void momentLimiterCUDA(int subDflg, int Nx, int Ny, int SubnElems, int nLoc, int pdeg, double *Q, double *QHalo);

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
  	int pdeg, nLoc, nNodes, nElems, SubnElems, nOrder1D, nOrder2D, Nx, Ny;
  	double T, dt, dx, dy;
  	double *w2d, *x2d, *w1d, *x1d, *phiVol, *dphixVol, *dphiyVol;
  	double *phiEdgeLeft, *phiEdgeRight, *phiEdgeBottom, *phiEdgeTop;

  	double *w2da, *x2da, *w1da, *x1da, *phiVola, *dphixVola, *dphiyVola;
  	double *phiEdgeLefta, *phiEdgeRighta, *phiEdgeBottoma, *phiEdgeTopa;
  	double *w2db, *x2db, *w1db, *x1db, *phiVolb, *dphixVolb, *dphiyVolb;
  	double *phiEdgeLeftb, *phiEdgeRightb, *phiEdgeBottomb, *phiEdgeTopb;

  	double *Q1, *Q2, *Q3, *Q4, *Q10, *Q20, *Q30, *Q40;
  	double *rhsQ1, *rhsQ2, *rhsQ3, *rhsQ4;
	double *Q1a, *Q1b, *Q2a, *Q2b, *Q3a, *Q3b, *Q4a, *Q4b;
	double *Q10a, *Q10b, *Q20a, *Q20b, *Q30a, *Q30b, *Q40a, *Q40b;
	double *rhsQ1a, *rhsQ1b, *rhsQ2a, *rhsQ2b, *rhsQ3a, *rhsQ3b, *rhsQ4a, *rhsQ4b;
	double *Q1aHalo, *Q2aHalo, *Q3aHalo, *Q4aHalo, *Q10aHalo, *Q20aHalo, *Q30aHalo, *Q40aHalo;
    double *Q1bHalo, *Q2bHalo, *Q3bHalo, *Q4bHalo, *Q10bHalo, *Q20bHalo, *Q30bHalo, *Q40bHalo;

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
	  SubnElems = nElems/2;

  	  Ny = dimy;

  	  // construct the mesh given by input parameter dim
  	  myMesh.quadConnectivity2D();

  	  nLoc = (pdeg+1)*(pdeg+1); // degrees of freedom per element
  	  nNodes = myMesh.getNumVerts();
      nElems = myMesh.getNumCells();
	  SubnElems = nElems/2;

      nOrder1D = pdeg+1; // number of quadrature nodes for surface integrals
      nOrder2D = (pdeg+2)*(pdeg+2); // number of quad nodes for volume integrals

      // initialize quadrature points
	  cudaMallocManaged(&w2d, nOrder2D*sizeof(double));
	  cudaMallocManaged(&x2d, nOrder2D*2*sizeof(double));
	  cudaMallocManaged(&w1d, nOrder1D*sizeof(double));
	  cudaMallocManaged(&x1d, nOrder1D*sizeof(double));

	  DGsolver::quadRule2D(sqrt(nOrder2D));
	  DGsolver::quadRule1D(nOrder1D);

	  // initialize basis functions in interior
	  cudaMallocManaged(&phiVol, nLoc*nOrder2D*sizeof(double));
	  cudaMallocManaged(&dphixVol, nLoc*nOrder2D*sizeof(double));
	  cudaMallocManaged(&dphiyVol, nLoc*nOrder2D*sizeof(double));
	  DGsolver::basisFunctions(pdeg, nOrder2D, phiVol, x2d);
	  DGsolver::basisFunctionsGrad(pdeg, nOrder2D, dphixVol, dphiyVol, x2d);

	  // initialize basis functions on edges
	  cudaMallocManaged(&phiEdgeLeft, nLoc*nOrder1D*sizeof(double));
	  cudaMallocManaged(&phiEdgeRight, nLoc*nOrder1D*sizeof(double));
	  cudaMallocManaged(&phiEdgeBottom, nLoc*nOrder1D*sizeof(double));
	  cudaMallocManaged(&phiEdgeTop, nLoc*nOrder1D*sizeof(double));
	  DGsolver::basisFunctionsEdge(pdeg);

	  // initialize solution arrays and right-hand side arrays
	  Q1 = new double[nElems*nLoc];
	  Q2 = new double[nElems*nLoc];
	  Q3 = new double[nElems*nLoc];
	  Q4 = new double[nElems*nLoc];

	  Q10 = new double[nElems*nLoc]();
	  Q20 = new double[nElems*nLoc]();
	  Q30 = new double[nElems*nLoc]();
	  Q40 = new double[nElems*nLoc]();

	  rhsQ1 = new double[nElems*nLoc];
	  rhsQ2 = new double[nElems*nLoc];
	  rhsQ3 = new double[nElems*nLoc];
	  rhsQ4 = new double[nElems*nLoc];

	  int can_access_peer_0_1, can_access_peer_1_0;
      cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, 1);
      cudaDeviceCanAccessPeer(&can_access_peer_1_0, 1, 0);
	  printf("0 can access 1? %s  1 can access 0? %s \n", can_access_peer_0_1 ? "true" : "false", can_access_peer_1_0 ? "true" : "false");
      
      cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0);
	  cudaSetDevice(1); cudaDeviceEnablePeerAccess(0,0);

	  cudaSetDevice(0);
	  cudaMalloc((void**) &Q1a, SubnElems*nLoc*sizeof(double));
	  cudaMalloc((void**) &Q2a, SubnElems*nLoc*sizeof(double));
	  cudaMalloc((void**) &Q3a, SubnElems*nLoc*sizeof(double));
	  cudaMalloc((void**) &Q4a, SubnElems*nLoc*sizeof(double));

	  cudaMalloc((void**) &Q10a, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q20a, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q30a, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q40a, SubnElems*nLoc*sizeof(double));

	  cudaMalloc((void**) &rhsQ1a, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &rhsQ2a, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &rhsQ3a, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &rhsQ4a, SubnElems*nLoc*sizeof(double));

	  cudaMalloc((void**) &Q1aHalo, Nx*nLoc*sizeof(double));
      cudaMalloc((void**) &Q2aHalo, Nx*nLoc*sizeof(double));
      cudaMalloc((void**) &Q3aHalo, Nx*nLoc*sizeof(double));
      cudaMalloc((void**) &Q4aHalo, Nx*nLoc*sizeof(double));

      cudaMalloc((void**) &phiVola, nLoc*nOrder2D*sizeof(double));
      cudaMalloc((void**) &dphixVola, nLoc*nOrder2D*sizeof(double));
      cudaMalloc((void**) &dphiyVola, nLoc*nOrder2D*sizeof(double));
      cudaMemcpy(phiVola, phiVol, nLoc*nOrder2D*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(dphixVola, dphixVol, nLoc*nOrder2D*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(dphiyVola, dphiyVol, nLoc*nOrder2D*sizeof(double), cudaMemcpyHostToDevice);

	  cudaMalloc((void**) &w2da, nOrder2D*sizeof(double));
	  cudaMalloc((void**) &x2da, nOrder2D*2*sizeof(double));
	  cudaMalloc((void**) &w1da, nOrder1D*sizeof(double));
	  cudaMalloc((void**) &x1da, nOrder1D*sizeof(double));
	  cudaMemcpy(w2da, w2d, nOrder2D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(x2da, x2d, nOrder2D*2*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(w1da, w1d, nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(x1da, x1d, nOrder1D*sizeof(double), cudaMemcpyHostToDevice);

	  cudaMalloc((void**) &phiEdgeLefta, nLoc*nOrder1D*sizeof(double));
	  cudaMalloc((void**) &phiEdgeRighta, nLoc*nOrder1D*sizeof(double));
	  cudaMalloc((void**) &phiEdgeBottoma, nLoc*nOrder1D*sizeof(double));
	  cudaMalloc((void**) &phiEdgeTopa, nLoc*nOrder1D*sizeof(double));
	  cudaMemcpy(phiEdgeLefta, phiEdgeLeft, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(phiEdgeRighta, phiEdgeRight, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(phiEdgeBottoma, phiEdgeBottom, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(phiEdgeTopa, phiEdgeTop, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);


	  cudaSetDevice(1);
	  cudaMalloc((void**) &Q1b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q2b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q3b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q4b, SubnElems*nLoc*sizeof(double));

	  cudaMalloc((void**) &Q10b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q20b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q30b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &Q40b, SubnElems*nLoc*sizeof(double));

	  cudaMalloc((void**) &rhsQ1b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &rhsQ2b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &rhsQ3b, SubnElems*nLoc*sizeof(double));
      cudaMalloc((void**) &rhsQ4b, SubnElems*nLoc*sizeof(double));

	  cudaMalloc((void**) &Q1bHalo, Nx*nLoc*sizeof(double));
	  cudaMalloc((void**) &Q2bHalo, Nx*nLoc*sizeof(double));
      cudaMalloc((void**) &Q3bHalo, Nx*nLoc*sizeof(double));
      cudaMalloc((void**) &Q4bHalo, Nx*nLoc*sizeof(double));

      cudaMalloc((void**) &phiVolb, nLoc*nOrder2D*sizeof(double));
      cudaMalloc((void**) &dphixVolb, nLoc*nOrder2D*sizeof(double));
      cudaMalloc((void**) &dphiyVolb, nLoc*nOrder2D*sizeof(double));
      cudaMemcpy(phiVolb, phiVol, nLoc*nOrder2D*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(dphixVolb, dphixVol, nLoc*nOrder2D*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(dphiyVolb, dphiyVol, nLoc*nOrder2D*sizeof(double), cudaMemcpyHostToDevice);

	  cudaMalloc((void**) &w2db, nOrder2D*sizeof(double));
	  cudaMalloc((void**) &x2db, nOrder2D*2*sizeof(double));
	  cudaMalloc((void**) &w1db, nOrder1D*sizeof(double));
	  cudaMalloc((void**) &x1db, nOrder1D*sizeof(double));
	  cudaMemcpy(w2db, w2d, nOrder2D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(x2db, x2d, nOrder2D*2*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(w1db, w1d, nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(x1db, x1d, nOrder1D*sizeof(double), cudaMemcpyHostToDevice);

	  cudaMalloc((void**) &phiEdgeLeftb, nLoc*nOrder1D*sizeof(double));
	  cudaMalloc((void**) &phiEdgeRightb, nLoc*nOrder1D*sizeof(double));
	  cudaMalloc((void**) &phiEdgeBottomb, nLoc*nOrder1D*sizeof(double));
	  cudaMalloc((void**) &phiEdgeTopb, nLoc*nOrder1D*sizeof(double));
	  cudaMemcpy(phiEdgeLeftb, phiEdgeLeft, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(phiEdgeRightb, phiEdgeRight, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(phiEdgeBottomb, phiEdgeBottom, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(phiEdgeTopb, phiEdgeTop, nLoc*nOrder1D*sizeof(double), cudaMemcpyHostToDevice);

	}

	// default destructor
	~DGsolver() {
	  delete[] Q1;
	  delete[] Q2;
	  delete[] Q3;
	  delete[] Q4;
	  delete[] Q10;
	  delete[] Q20;
	  delete[] Q30;
	  delete[] Q40;
	  delete[] rhsQ1;
	  delete[] rhsQ2;
	  delete[] rhsQ3;
	  delete[] rhsQ4;

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

	  cudaSetDevice(0);
	  cudaFree(Q1a); cudaFree(Q2a); cudaFree(Q3a); cudaFree(Q4a);
	  cudaFree(Q10a); cudaFree(Q20a); cudaFree(Q30a); cudaFree(Q40a);
	  cudaFree(rhsQ1a); cudaFree(rhsQ2a); cudaFree(rhsQ3a); cudaFree(rhsQ4a);
	  cudaFree(Q1aHalo); cudaFree(Q2aHalo); cudaFree(Q3aHalo); cudaFree(Q4aHalo);
	  cudaFree(phiVola); cudaFree(dphixVola); cudaFree(dphiyVola);
	  cudaFree(w2da); cudaFree(x2da); cudaFree(w1da); cudaFree(x1da);
	  cudaFree(phiEdgeLefta); cudaFree(phiEdgeRighta); cudaFree(phiEdgeBottoma); cudaFree(phiEdgeTopa);

	  cudaSetDevice(1);
	  cudaFree(Q1b); cudaFree(Q2b); cudaFree(Q3b); cudaFree(Q4b);
      cudaFree(Q10b); cudaFree(Q20b); cudaFree(Q30b); cudaFree(Q40b);
      cudaFree(rhsQ1b); cudaFree(rhsQ2b); cudaFree(rhsQ3b); cudaFree(rhsQ4b);
      cudaFree(Q1bHalo); cudaFree(Q2bHalo); cudaFree(Q3bHalo); cudaFree(Q4bHalo);
      cudaFree(phiVolb); cudaFree(dphixVolb); cudaFree(dphiyVolb);
	  cudaFree(w2db); cudaFree(x2db); cudaFree(w1db); cudaFree(x1db);
	  cudaFree(phiEdgeLeftb); cudaFree(phiEdgeRightb); cudaFree(phiEdgeBottomb); cudaFree(phiEdgeTopb);
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
	void runCUDAmomentLimiter(double *Qa, double *Qb, double *QaHalo, double *QbHalo);

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

	// zero stage: copy initial data into Qa and Qb
	cudaSetDevice(0); //copy first half of array (bottom) to device zero
	cudaMemcpy(Q10a,Q10,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q20a,Q20,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q30a,Q30,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q40a,Q40,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(Q1a,Q10,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q2a,Q20,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q3a,Q30,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q4a,Q40,SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);

	// get bottom row of upper subdomain
	cudaMemcpy(Q1aHalo, Q10+nLoc*SubnElems, Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q2aHalo, Q20+nLoc*SubnElems, Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q3aHalo, Q30+nLoc*SubnElems, Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Q4aHalo, Q40+nLoc*SubnElems, Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);

	cudaSetDevice(1); //copy second half of array (top) to device one
    cudaMemcpy(Q10b,Q10+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q20b,Q20+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q30b,Q30+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q40b,Q40+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(Q1b,Q10+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q2b,Q20+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q3b,Q30+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q4b,Q40+nLoc*SubnElems, SubnElems*nLoc*sizeof(double), cudaMemcpyHostToDevice);

	// get top row of lower subdomain
    cudaMemcpy(Q1bHalo, Q10+nLoc*(SubnElems-Nx), Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q2bHalo, Q20+nLoc*(SubnElems-Nx), Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q3bHalo, Q30+nLoc*(SubnElems-Nx), Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Q4bHalo, Q40+nLoc*(SubnElems-Nx), Nx*nLoc*sizeof(double), cudaMemcpyHostToDevice);

    double t1 = omp_get_wtime();


	int iter = 0;
	int plotiter = 0;
	while (t < T) {
		// first RK stage
		DGsolver::runCUDAcomputeRHS();
		DGsolver::runCUDAupdate1();
		
		//update Halos
		cudaSetDevice(0); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1aHalo, 0, Q1b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2aHalo, 0, Q2b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3aHalo, 0, Q3b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4aHalo, 0, Q4b, 1, Nx*nLoc*sizeof(double));

		cudaSetDevice(1); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1bHalo, 1, Q1a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2bHalo, 1, Q2a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3bHalo, 1, Q3a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4bHalo, 1, Q4a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));

		DGsolver::runCUDAmomentLimiter(Q1a, Q1b, Q1aHalo, Q1bHalo);
		DGsolver::runCUDAmomentLimiter(Q2a, Q2b, Q2aHalo, Q2bHalo);
		DGsolver::runCUDAmomentLimiter(Q3a, Q3b, Q3aHalo, Q3bHalo);
		DGsolver::runCUDAmomentLimiter(Q4a, Q4b, Q4aHalo, Q4bHalo);

		//update Halos after applying limiter
		cudaSetDevice(0); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1aHalo, 0, Q1b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2aHalo, 0, Q2b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3aHalo, 0, Q3b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4aHalo, 0, Q4b, 1, Nx*nLoc*sizeof(double));

        cudaSetDevice(1); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1bHalo, 1, Q1a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2bHalo, 1, Q2a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3bHalo, 1, Q3a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4bHalo, 1, Q4a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));


		// second RK stage
		DGsolver::runCUDAcomputeRHS();
		DGsolver::runCUDAupdate2();

		//update Halos
		cudaSetDevice(0); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1aHalo, 0, Q1b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2aHalo, 0, Q2b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3aHalo, 0, Q3b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4aHalo, 0, Q4b, 1, Nx*nLoc*sizeof(double));

        cudaSetDevice(1); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1bHalo, 1, Q1a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2bHalo, 1, Q2a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3bHalo, 1, Q3a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4bHalo, 1, Q4a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));


		// apply moment limiter
        DGsolver::runCUDAmomentLimiter(Q1a, Q1b, Q1aHalo, Q1bHalo);
		DGsolver::runCUDAmomentLimiter(Q2a, Q2b, Q2aHalo, Q2bHalo);
		DGsolver::runCUDAmomentLimiter(Q3a, Q3b, Q3aHalo, Q3bHalo);
		DGsolver::runCUDAmomentLimiter(Q4a, Q4b, Q4aHalo, Q4bHalo);

		//update Halos after applying limiter
		cudaSetDevice(0); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1aHalo, 0, Q1b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2aHalo, 0, Q2b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3aHalo, 0, Q3b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4aHalo, 0, Q4b, 1, Nx*nLoc*sizeof(double));

        cudaSetDevice(1); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1bHalo, 1, Q1a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2bHalo, 1, Q2a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3bHalo, 1, Q3a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4bHalo, 1, Q4a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));

		// third RK stage
		DGsolver::runCUDAcomputeRHS();
		DGsolver::runCUDAupdate3();

        //update Halos
    	cudaSetDevice(0); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1aHalo, 0, Q10b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2aHalo, 0, Q20b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3aHalo, 0, Q30b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4aHalo, 0, Q40b, 1, Nx*nLoc*sizeof(double));

        cudaSetDevice(1); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1bHalo, 1, Q10a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2bHalo, 1, Q20a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3bHalo, 1, Q30a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4bHalo, 1, Q40a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
		

		// apply moment limiter component-wise
		DGsolver::runCUDAmomentLimiter(Q10a, Q10b, Q1aHalo, Q1bHalo);
		DGsolver::runCUDAmomentLimiter(Q20a, Q20b, Q2aHalo, Q2bHalo);
		DGsolver::runCUDAmomentLimiter(Q30a, Q30b, Q3aHalo, Q3bHalo);
		DGsolver::runCUDAmomentLimiter(Q40a, Q40b, Q4aHalo, Q4bHalo);

		//update Halos
		cudaSetDevice(0); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1aHalo, 0, Q10b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2aHalo, 0, Q20b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3aHalo, 0, Q30b, 1, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4aHalo, 0, Q40b, 1, Nx*nLoc*sizeof(double));

        cudaSetDevice(1); cudaDeviceSynchronize();
        cudaMemcpyPeer(Q1bHalo, 1, Q10a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q2bHalo, 1, Q20a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q3bHalo, 1, Q30a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));
        cudaMemcpyPeer(Q4bHalo, 1, Q40a+nLoc*(SubnElems-Nx), 0, Nx*nLoc*sizeof(double));

		// zero stage: copy initial data to Q
		DGsolver::runCUDAupdate4();
		cudaSetDevice(0); cudaDeviceSynchronize();
		cudaSetDevice(1); cudaDeviceSynchronize();
		
		// // dump density solution to .vtk output every 20 time steps
		// if (iter%300 == 0)
		// {
		// 	cudaSetDevice(0); cudaDeviceSynchronize();
		// 	cudaMemcpy(Q10, Q10a, SubnElems*nLoc*sizeof(double), cudaMemcpyDeviceToHost);

		// 	cudaSetDevice(1); cudaDeviceSynchronize();
		// 	cudaMemcpy(Q10+nLoc*SubnElems, Q10b, SubnElems*nLoc*sizeof(double), cudaMemcpyDeviceToHost);


		// 	double *uCell = new double[nOrder2D];
		// 	double *sol = new double[nElems]();
		// 	for (int k = 0; k < nElems; ++k)
		// 	{
		// 		// compute the solution at the 2D quadrature nodes in each element
		// 		for (int i = 0; i < nOrder2D; ++i)
		// 		{
		// 			uCell[i] = 0.0;
		// 			for (int j = 0; j < nLoc; ++j)
		// 			{
		// 				uCell[i] += Q10[k*nLoc+j]*phiVol[j*nOrder2D+i];
		// 			}			
		// 		}

		// 		// take the average of the solution in the element and assign it
		// 		// to the field value in the .vtk file
		// 		for (int i = 0; i < nOrder2D; ++i)
		// 		{
		// 			sol[k] += uCell[i]/nOrder2D;
		// 		}
		// 	}

		// 	delete[] uCell;

		// 	writeVTK<double>(nElems, nNodes, myMesh.getEToV(), myMesh.getVxy(), sol, t, plotiter);
		// 	plotiter++;
		// }

		printf("Time: %f\n", t);
		t += dt;
		iter++;
	}

	cudaSetDevice(0); cudaDeviceSynchronize();
	cudaSetDevice(1); cudaDeviceSynchronize();
	double t2 = omp_get_wtime();

	printf("Elapsed time for time stepping: %f\n", t2-t1);

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

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeRHScuda, 0, SubnElems);
    int numBlocks = (SubnElems + blockSize - 1)/(blockSize);

    int nGPU = 2;
    #pragma omp parallel for
    for (int n = 0; n < nGPU; ++n)
    {
    	cudaSetDevice(n);
    	if (n == 0)
    	{
    		computeRHScuda<<<numBlocks, blockSize>>>(n, Nx, Ny, SubnElems, nOrder2D, nOrder1D, nLoc, Q1a, Q2a, Q3a, Q4a, Q1aHalo, Q2aHalo, Q3aHalo, Q4aHalo, 
									rhsQ1a, rhsQ2a, rhsQ3a, rhsQ4a, phiVola, dphixVola, dphiyVola, w2da, w1da,
									phiEdgeLefta, phiEdgeRighta, phiEdgeBottoma, phiEdgeTopa);
    	} else {
    		computeRHScuda<<<numBlocks, blockSize>>>(n, Nx, Ny, SubnElems, nOrder2D, nOrder1D, nLoc, Q1b, Q2b, Q3b, Q4b, Q1bHalo, Q2bHalo, Q3bHalo, Q4bHalo, 
                                            rhsQ1b, rhsQ2b, rhsQ3b, rhsQ4b, phiVolb, dphixVolb, dphiyVolb, w2db, w1db,
                                            phiEdgeLeftb, phiEdgeRightb, phiEdgeBottomb, phiEdgeTopb);
    	}
    }

}


inline void DGsolver::runCUDAupdate1() {
	int blockSize;
	int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA1, 0, SubnElems*nLoc);
    int numBlocks = (SubnElems*nLoc + blockSize - 1)/(blockSize);

    int nGPU = 2;
    #pragma omp parallel for
    for (int n = 0; n < nGPU; ++n)
    {
    	cudaSetDevice(n);
    	if (n == 0)
    	{
    		updateCUDA1<<<numBlocks, blockSize>>>(SubnElems, nLoc, dt, Q1a, Q2a, Q3a, Q4a, Q10a, Q20a, Q30a, Q40a, rhsQ1a, rhsQ2a, rhsQ3a, rhsQ4a);
    	} else {
    		updateCUDA1<<<numBlocks, blockSize>>>(SubnElems, nLoc, dt, Q1b, Q2b, Q3b, Q4b, Q10b, Q20b, Q30b, Q40b, rhsQ1b, rhsQ2b, rhsQ3b, rhsQ4b);
    	}
    }
}


inline void DGsolver::runCUDAupdate2() {
	int blockSize;
	int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA2, 0, SubnElems*nLoc);
    int numBlocks = (SubnElems*nLoc + blockSize - 1)/(blockSize);

    int nGPU = 2;
    #pragma omp parallel for
    for (int n = 0; n < nGPU; ++n)
    {
    	cudaSetDevice(n);
    	if (n == 0)
    	{
    		updateCUDA2<<<numBlocks, blockSize>>>(SubnElems, nLoc, dt, Q1a, Q2a, Q3a, Q4a, Q10a, Q20a, Q30a, Q40a, rhsQ1a, rhsQ2a, rhsQ3a, rhsQ4a);
    	} else {
    		updateCUDA2<<<numBlocks, blockSize>>>(SubnElems, nLoc, dt, Q1b, Q2b, Q3b, Q4b, Q10b, Q20b, Q30b, Q40b, rhsQ1b, rhsQ2b, rhsQ3b, rhsQ4b);
    	}
    }
}


inline void DGsolver::runCUDAupdate3() {
	int blockSize;
	int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA3, 0, SubnElems*nLoc);
    int numBlocks = (SubnElems*nLoc + blockSize - 1)/(blockSize);

    int nGPU = 2;
    #pragma omp parallel for
    for (int n = 0; n < nGPU; ++n)
    {
    	cudaSetDevice(n);
    	if (n == 0)
    	{
    		updateCUDA3<<<numBlocks, blockSize>>>(SubnElems, nLoc, dt, Q1a, Q2a, Q3a, Q4a, Q10a, Q20a, Q30a, Q40a, rhsQ1a, rhsQ2a, rhsQ3a, rhsQ4a);
    	} else {
    		updateCUDA3<<<numBlocks, blockSize>>>(SubnElems, nLoc, dt, Q1b, Q2b, Q3b, Q4b, Q10b, Q20b, Q30b, Q40b, rhsQ1b, rhsQ2b, rhsQ3b, rhsQ4b);
    	}
    }
}


inline void DGsolver::runCUDAupdate4() {
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateCUDA4, 0, SubnElems*nLoc);
	int numBlocks = (SubnElems*nLoc + blockSize - 1)/(blockSize);

	int nGPU = 2;
    #pragma omp parallel for
    for (int n = 0; n < nGPU; ++n)
    {
    	cudaSetDevice(n);
    	if (n == 0)
    	{
    		updateCUDA4<<<numBlocks, blockSize>>>(SubnElems, nLoc, Q1a, Q2a, Q3a, Q4a, Q10a, Q20a, Q30a, Q40a);
    	} else {
    		updateCUDA4<<<numBlocks, blockSize>>>(SubnElems, nLoc, Q1b, Q2b, Q3b, Q4b, Q10b, Q20b, Q30b, Q40b);
    	}
    }
}


inline void DGsolver::runCUDAmomentLimiter(double *Qa, double *Qb, double *QaHalo, double *QbHalo) {
	int blockSize;
	int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, momentLimiterCUDA, 0, SubnElems);
    int numBlocks = (SubnElems + blockSize - 1)/(blockSize);
	
	int nGPU = 2;
    #pragma omp parallel for
    for (int n = 0; n < nGPU; ++n)
    {
    	cudaSetDevice(n);
    	if (n == 0)
    	{
    		momentLimiterCUDA<<<numBlocks, blockSize>>>(0, Nx, Ny, SubnElems, nLoc, pdeg, Qa, QaHalo);
    	} else {
    		momentLimiterCUDA<<<numBlocks, blockSize>>>(1, Nx, Ny, SubnElems, nLoc, pdeg, Qb, QbHalo);
    	}
    }
}

#include "../cuda/DGsolveCUDA.cu"

#endif

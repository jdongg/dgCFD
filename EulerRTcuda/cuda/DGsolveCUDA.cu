#include "DGsolve.hpp"
#include "meshing.hpp"

class DGsolver;

__device__
void volumeFluxF(int N, double *Q, double *F);

__device__
void volumeFluxG(int N, double *Q, double *G);

__device__
void numericalFluxF(int nOrder1D, double *QLeft, double *QRight, double *Fh);

__device__
void numericalFluxG(int nOrder1D, double *QLeft, double *QRight, double *Gh);

__global__
void computeRHScuda(int Nx, int Ny, int nElems, int nOrder2D, int nOrder1D, int nLoc, double *Vxy, int *EToV, int *mapB, double *Q1, double *Q2, 
					double *Q3, double *Q4, double *rhsQ1, double *rhsQ2, double *rhsQ3, double *rhsQ4,
				    double *phiVol, double *dphixVol, double *dphiyVol, double *w2d, double *w1d,
				    double *phiEdgeLeft, double *phiEdgeRight, double *phiEdgeBottom, double *phiEdgeTop) {

	// we compute the RHS contributions in parallel on each element;
	// each thread is assigned a different element to work on. with OpenMP
	// shared memory, the variables xa, xb, ya, yb, vol, and the local
	// arrays to store neighbor coefficients and solution values must 
	// be intialized with the scope of the main for loop over nElems, otherwise
	// we will have a race condition. 
	int gtid = threadIdx.x + blockIdx.x*blockDim.x;
	for (int k = gtid; k < nElems; k += gridDim.x*blockDim.x) {
		// grab coordinates of vertices on element k
		double xa, xb, ya, yb, vol;

		xa = Vxy[EToV[k*6]*2];
		xb = Vxy[EToV[k*6+1]*2];
		ya = Vxy[EToV[k*6]*2+1];
		yb = Vxy[EToV[k*6+2]*2+1];

		// Jacobian of transformation from reference element to
		// element k
		vol = (xb-xa)*(yb-ya)/4.0;

		// double *c = new double[4*nLoc];
		// double *Qloc = new double[4*nOrder2D];
		// double *F = new double[4*nOrder2D];
		// double *G = new double[4*nOrder2D];
		// double *Fh = new double[4*nOrder1D];
		// double *Gh = new double[4*nOrder1D];

		// hard code these arrays into stack memory for now
		double c[4*9];
		double Qloc[4*16];
		double F[4*16];
		double G[4*16];
		double Fh[4*3];
		double Gh[4*3];

		// extract coefficients on cell k
		for (int i = 0; i < nLoc; ++i)
		{
			c[i] = Q1[k*nLoc+i];
			c[nLoc+i] = Q2[k*nLoc+i];
			c[2*nLoc+i] = Q3[k*nLoc+i];
			c[3*nLoc+i] = Q4[k*nLoc+i];
		}

		// compute conserved variables on element k
		for (int i = 0; i < nOrder2D; ++i)
		{
			Qloc[i] = 0.0;
			Qloc[nOrder2D+i] = 0.0;
			Qloc[2*nOrder2D+i] = 0.0;
			Qloc[3*nOrder2D+i] = 0.0;

			for (int j = 0; j < nLoc; ++j)
			{
				Qloc[i] += c[j]*phiVol[j*nOrder2D+i];
				Qloc[nOrder2D+i] += c[nLoc+j]*phiVol[j*nOrder2D+i];
				Qloc[2*nOrder2D+i] += c[2*nLoc+j]*phiVol[j*nOrder2D+i];
				Qloc[3*nOrder2D+i] += c[3*nLoc+j]*phiVol[j*nOrder2D+i];
			}
		}

		// compute volume fluxes on element k
		volumeFluxF(nOrder2D, Qloc, F);
		volumeFluxG(nOrder2D, Qloc, G);

		// compute volume contributions on element k
		for (int i = 0; i < nLoc; ++i)
		{
			int idx = k*nLoc+i;

			rhsQ1[idx] = 0.0;
			rhsQ2[idx] = 0.0;
			rhsQ3[idx] = 0.0;
			rhsQ4[idx] = 0.0;

			for (int j = 0; j < nOrder2D; ++j)
			{
				int jdx = i*nOrder2D+j;
				rhsQ1[idx] += vol*w2d[j]*(dphixVol[jdx]*F[j] + dphiyVol[jdx]*G[j]);
				rhsQ2[idx] += vol*w2d[j]*(dphixVol[jdx]*F[nOrder2D+j] + dphiyVol[jdx]*G[nOrder2D+j]);
				rhsQ3[idx] += vol*w2d[j]*(dphixVol[jdx]*F[2*nOrder2D+j] + dphiyVol[jdx]*G[2*nOrder2D+j]);
				rhsQ4[idx] += vol*w2d[j]*(dphixVol[jdx]*F[3*nOrder2D+j] + dphiyVol[jdx]*G[3*nOrder2D+j]);

				// gravity source terms for Rayleigh-Taylor instability
				rhsQ3[idx] += vol*w2d[j]*Qloc[j]*phiVol[jdx];
				rhsQ4[idx] += vol*w2d[j]*Qloc[2*nOrder2D+j]*phiVol[jdx];
			}
		}


		// determine neighbor cells
		int idx = EToV[k*6+4];
		int jdx = EToV[k*6+5];

		int nLeft = mapB[k*4];
		int nRight = mapB[k*4+1];
		int nBottom = mapB[k*4+2];
		int nTop = mapB[k*4+3];

		// grab basis coefficients on neighbor cells
		// double *cLeft = new double[nLoc*4];
		// double *cRight = new double[nLoc*4];
		// double *cBottom = new double[nLoc*4];
		// double *cTop = new double[nLoc*4];

		double cLeft[9*4];
		double cRight[9*4];
		double cBottom[9*4];
		double cTop[9*4];

		for (int i = 0; i < nLoc; ++i)
		{
			cLeft[i] = Q1[nLeft*nLoc+i];
			cLeft[nLoc+i] = Q2[nLeft*nLoc+i];
			cLeft[2*nLoc+i] = Q3[nLeft*nLoc+i];
			cLeft[3*nLoc+i] = Q4[nLeft*nLoc+i];

			cRight[i] = Q1[nRight*nLoc+i];
			cRight[nLoc+i] = Q2[nRight*nLoc+i];
			cRight[2*nLoc+i] = Q3[nRight*nLoc+i];
			cRight[3*nLoc+i] = Q4[nRight*nLoc+i];

			cBottom[i] = Q1[nBottom*nLoc+i];
			cBottom[nLoc+i] = Q2[nBottom*nLoc+i];
			cBottom[2*nLoc+i] = Q3[nBottom*nLoc+i];
			cBottom[3*nLoc+i] = Q4[nBottom*nLoc+i];

			cTop[i] = Q1[nTop*nLoc+i];
			cTop[nLoc+i] = Q2[nTop*nLoc+i];
			cTop[2*nLoc+i] = Q3[nTop*nLoc+i];
			cTop[3*nLoc+i] = Q4[nTop*nLoc+i];
		}

		// double *Qedge = new double[4*nOrder1D];
		// double *Qneighbor = new double[4*nOrder1D];

		double Qedge[4*3];
		double Qneighbor[4*3];

		// left edge
		for (int i = 0; i < nOrder1D; ++i)
		{
			Qedge[i] = 0.0;
			Qedge[nOrder1D+i] = 0.0;
			Qedge[2*nOrder1D+i] = 0.0;
			Qedge[3*nOrder1D+i] = 0.0;

			Qneighbor[i] = 0.0;
			Qneighbor[nOrder1D+i] = 0.0;
			Qneighbor[2*nOrder1D+i] = 0.0;
			Qneighbor[3*nOrder1D+i] = 0.0;

			for (int j = 0; j < nLoc; ++j)
			{
				Qedge[i] += c[j]*phiEdgeLeft[j*nOrder1D+i];
				Qedge[nOrder1D+i] += c[nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
				Qedge[2*nOrder1D+i] += c[2*nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
				Qedge[3*nOrder1D+i] += c[3*nLoc+j]*phiEdgeLeft[j*nOrder1D+i];

				// if (jdx > 0)
				// {
					Qneighbor[i] += cLeft[j]*phiEdgeRight[j*nOrder1D+i];
					Qneighbor[nOrder1D+i] += cLeft[nLoc+j]*phiEdgeRight[j*nOrder1D+i];
					Qneighbor[2*nOrder1D+i] += cLeft[2*nLoc+j]*phiEdgeRight[j*nOrder1D+i];
					Qneighbor[3*nOrder1D+i] += cLeft[3*nLoc+j]*phiEdgeRight[j*nOrder1D+i];
				// }
				// else { // enforce natural BCs on the left boundary; left neighbor is itself
				// 	Qneighbor[i] += c[j]*phiEdgeLeft[j*nOrder1D+i];
				// 	Qneighbor[nOrder1D+i] += c[nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
				// 	Qneighbor[2*nOrder1D+i] += c[2*nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
				// 	Qneighbor[3*nOrder1D+i] += c[3*nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
				// }
			}
		}

		// compute numerical flux along left edge
		double vol1d = (xb-xa)/2.0;
		numericalFluxF(nOrder1D, Qedge, Qneighbor, Fh);

		for (int i = 0; i < nLoc; ++i)
		{
			for (int j = 0; j < nOrder1D; ++j)
			{
				rhsQ1[k*nLoc+i] += vol1d*w1d[j] * phiEdgeLeft[i*nOrder1D+j] * Fh[j];
				rhsQ2[k*nLoc+i] += vol1d*w1d[j] * phiEdgeLeft[i*nOrder1D+j] * Fh[nOrder1D+j];
				rhsQ3[k*nLoc+i] += vol1d*w1d[j] * phiEdgeLeft[i*nOrder1D+j] * Fh[2*nOrder1D+j];
				rhsQ4[k*nLoc+i] += vol1d*w1d[j] * phiEdgeLeft[i*nOrder1D+j] * Fh[3*nOrder1D+j];
			}
		}


		// bottom neighbor
		for (int i = 0; i < nOrder1D; ++i)
		{
			Qedge[i] = 0.0;
			Qedge[nOrder1D+i] = 0.0;
			Qedge[2*nOrder1D+i] = 0.0;
			Qedge[3*nOrder1D+i] = 0.0;

			Qneighbor[i] = 0.0;
			Qneighbor[nOrder1D+i] = 0.0;
			Qneighbor[2*nOrder1D+i] = 0.0;
			Qneighbor[3*nOrder1D+i] = 0.0;

			for (int j = 0; j < nLoc; ++j)
			{
				Qedge[i] += c[j]*phiEdgeBottom[j*nOrder1D+i];
				Qedge[nOrder1D+i] += c[nLoc+j]*phiEdgeBottom[j*nOrder1D+i];
				Qedge[2*nOrder1D+i] += c[2*nLoc+j]*phiEdgeBottom[j*nOrder1D+i];
				Qedge[3*nOrder1D+i] += c[3*nLoc+j]*phiEdgeBottom[j*nOrder1D+i];

				if (idx > 0)
				{
					Qneighbor[i] += cBottom[j]*phiEdgeTop[j*nOrder1D+i];
					Qneighbor[nOrder1D+i] += cBottom[nLoc+j]*phiEdgeTop[j*nOrder1D+i];
					Qneighbor[2*nOrder1D+i] += cBottom[2*nLoc+j]*phiEdgeTop[j*nOrder1D+i];
					Qneighbor[3*nOrder1D+i] += cBottom[3*nLoc+j]*phiEdgeTop[j*nOrder1D+i];
				}
				else { // enforce Dirichlet BCs on the bottom boundary
					Qneighbor[i] = 2.0;
					Qneighbor[nOrder1D+i] = 0.0;
					Qneighbor[2*nOrder1D+i] = 0.0;
					Qneighbor[3*nOrder1D+i] = 1.0/0.4;
				}
			}
		}

		// compute numerical flux along bottom boundary
		numericalFluxG(nOrder1D, Qedge, Qneighbor, Gh);

		for (int i = 0; i < nLoc; ++i)
		{
			for (int j = 0; j < nOrder1D; ++j)
			{
				rhsQ1[k*nLoc+i] += vol1d*w1d[j] * phiEdgeBottom[i*nOrder1D+j] * Gh[j];
				rhsQ2[k*nLoc+i] += vol1d*w1d[j] * phiEdgeBottom[i*nOrder1D+j] * Gh[nOrder1D+j];
				rhsQ3[k*nLoc+i] += vol1d*w1d[j] * phiEdgeBottom[i*nOrder1D+j] * Gh[2*nOrder1D+j];
				rhsQ4[k*nLoc+i] += vol1d*w1d[j] * phiEdgeBottom[i*nOrder1D+j] * Gh[3*nOrder1D+j];
			}
		}


		// right edge
		for (int i = 0; i < nOrder1D; ++i)
		{
			Qedge[i] = 0.0;
			Qedge[nOrder1D+i] = 0.0;
			Qedge[2*nOrder1D+i] = 0.0;
			Qedge[3*nOrder1D+i] = 0.0;

			Qneighbor[i] = 0.0;
			Qneighbor[nOrder1D+i] = 0.0;
			Qneighbor[2*nOrder1D+i] = 0.0;
			Qneighbor[3*nOrder1D+i] = 0.0;

			for (int j = 0; j < nLoc; ++j)
			{
				Qedge[i] += c[j]*phiEdgeRight[j*nOrder1D+i];
				Qedge[nOrder1D+i] += c[nLoc+j]*phiEdgeRight[j*nOrder1D+i];
				Qedge[2*nOrder1D+i] += c[2*nLoc+j]*phiEdgeRight[j*nOrder1D+i];
				Qedge[3*nOrder1D+i] += c[3*nLoc+j]*phiEdgeRight[j*nOrder1D+i];

				// if (jdx < Nx-1)
				// {
					Qneighbor[i] += cRight[j]*phiEdgeLeft[j*nOrder1D+i];
					Qneighbor[nOrder1D+i] += cRight[nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
					Qneighbor[2*nOrder1D+i] += cRight[2*nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
					Qneighbor[3*nOrder1D+i] += cRight[3*nLoc+j]*phiEdgeLeft[j*nOrder1D+i];
				// }
				// else { // enforce natural BCs along right edge; right neighbor is itself
				// 	Qneighbor[i] += c[j]*phiEdgeRight[j*nOrder1D+i];
				// 	Qneighbor[nOrder1D+i] += c[nLoc+j]*phiEdgeRight[j*nOrder1D+i];
				// 	Qneighbor[2*nOrder1D+i] += c[2*nLoc+j]*phiEdgeRight[j*nOrder1D+i];
				// 	Qneighbor[3*nOrder1D+i] += c[3*nLoc+j]*phiEdgeRight[j*nOrder1D+i];
				// }
			}
		}

		// compute numerical flux along right edge
		numericalFluxF(nOrder1D, Qneighbor, Qedge, Fh);

		for (int i = 0; i < nLoc; ++i)
		{
			for (int j = 0; j < nOrder1D; ++j)
			{
				rhsQ1[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeRight[i*nOrder1D+j] * Fh[j];
				rhsQ2[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeRight[i*nOrder1D+j] * Fh[nOrder1D+j];
				rhsQ3[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeRight[i*nOrder1D+j] * Fh[2*nOrder1D+j];
				rhsQ4[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeRight[i*nOrder1D+j] * Fh[3*nOrder1D+j];
			}
		}


		// top edge
		for (int i = 0; i < nOrder1D; ++i)
		{
			Qedge[i] = 0.0;
			Qedge[nOrder1D+i] = 0.0;
			Qedge[2*nOrder1D+i] = 0.0;
			Qedge[3*nOrder1D+i] = 0.0;

			Qneighbor[i] = 0.0;
			Qneighbor[nOrder1D+i] = 0.0;
			Qneighbor[2*nOrder1D+i] = 0.0;
			Qneighbor[3*nOrder1D+i] = 0.0;

			for (int j = 0; j < nLoc; ++j)
			{
				Qedge[i] += c[j]*phiEdgeTop[j*nOrder1D+i];
				Qedge[nOrder1D+i] += c[nLoc+j]*phiEdgeTop[j*nOrder1D+i];
				Qedge[2*nOrder1D+i] += c[2*nLoc+j]*phiEdgeTop[j*nOrder1D+i];
				Qedge[3*nOrder1D+i] += c[3*nLoc+j]*phiEdgeTop[j*nOrder1D+i];

				if (idx < Ny-1)
				{
					Qneighbor[i] += cTop[j]*phiEdgeBottom[j*nOrder1D+i];
					Qneighbor[nOrder1D+i] += cTop[nLoc+j]*phiEdgeBottom[j*nOrder1D+i];
					Qneighbor[2*nOrder1D+i] += cTop[2*nLoc+j]*phiEdgeBottom[j*nOrder1D+i];
					Qneighbor[3*nOrder1D+i] += cTop[3*nLoc+j]*phiEdgeBottom[j*nOrder1D+i];
				}
				else { // enforce Dirichlet BCs along top edge
					Qneighbor[i] = 1.0;
					Qneighbor[nOrder1D+i] = 0.0;
					Qneighbor[2*nOrder1D+i] = 0.0;
					Qneighbor[3*nOrder1D+i] = 2.5/0.4;
				}
			}
		}

		// compute numerical flux along top edge
		numericalFluxG(nOrder1D, Qneighbor, Qedge, Gh);

		for (int i = 0; i < nLoc; ++i)
		{
			for (int j = 0; j < nOrder1D; ++j)
			{
				rhsQ1[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeTop[i*nOrder1D+j] * Gh[j];
				rhsQ2[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeTop[i*nOrder1D+j] * Gh[nOrder1D+j];
				rhsQ3[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeTop[i*nOrder1D+j] * Gh[2*nOrder1D+j];
				rhsQ4[k*nLoc+i] -= vol1d*w1d[j] * phiEdgeTop[i*nOrder1D+j] * Gh[3*nOrder1D+j];
			}
		}

		// we need to scale by the Jacobian because the fully-discrete scheme is
		//
		//		Mk*(u^{n+1}-u^{n}) = dt*L(u^{n}),
		//
		// where Mk is the local mass matrix. Since we are using the Legendre basis 
		// (i.e. an orthonormal basis), we have Mk = vol*Ik, where Ik is the identity
		// matrix. 
		for (int i = 0; i < nLoc; ++i)
		{
			rhsQ1[k*nLoc+i] = rhsQ1[k*nLoc+i]/vol;
			rhsQ2[k*nLoc+i] = rhsQ2[k*nLoc+i]/vol;
			rhsQ3[k*nLoc+i] = rhsQ3[k*nLoc+i]/vol;
			rhsQ4[k*nLoc+i] = rhsQ4[k*nLoc+i]/vol;
		}

		// free up local memory
		// delete[] c;
		// delete[] Qloc;
		// delete[] F;
		// delete[] G;
		// delete[] Fh;
		// delete[] Gh;
		// delete[] cLeft;
		// delete[] cRight;
		// delete[] cBottom;
		// delete[] cTop;
		// delete[] Qedge;
		// delete[] Qneighbor;

	}
}


__global__
void updateCUDA1(int nElems, int nLoc, double dt, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40, double *rhsQ1, double *rhsQ2,
	             double *rhsQ3, double *rhsQ4) {
	int gtid = threadIdx.x + blockIdx.x*blockDim.x;
	for (int i = gtid; i < nElems*nLoc; i += gridDim.x*blockDim.x)
	{
		Q1[i] = Q10[i] + dt*rhsQ1[i];
		Q2[i] = Q20[i] + dt*rhsQ2[i];
		Q3[i] = Q30[i] + dt*rhsQ3[i];
		Q4[i] = Q40[i] + dt*rhsQ4[i];
	}
}


__global__
void updateCUDA2(int nElems, int nLoc, double dt, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40, double *rhsQ1, double *rhsQ2,
	             double *rhsQ3, double *rhsQ4) {
	int gtid = threadIdx.x + blockIdx.x*blockDim.x;
	for (int i = gtid; i < nElems*nLoc; i += gridDim.x*blockDim.x)
	{
		Q1[i] = 3.0/4.0*Q10[i] + (Q1[i] + dt*rhsQ1[i])/4.0;
		Q2[i] = 3.0/4.0*Q20[i] + (Q2[i] + dt*rhsQ2[i])/4.0;
		Q3[i] = 3.0/4.0*Q30[i] + (Q3[i] + dt*rhsQ3[i])/4.0;
		Q4[i] = 3.0/4.0*Q40[i] + (Q4[i] + dt*rhsQ4[i])/4.0;
	}
}


__global__
void updateCUDA3(int nElems, int nLoc, double dt, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40, double *rhsQ1, double *rhsQ2,
	             double *rhsQ3, double *rhsQ4) {
	int gtid = threadIdx.x + blockIdx.x*blockDim.x;
	for (int i = gtid; i < nElems*nLoc; i += gridDim.x*blockDim.x)
	{
		Q10[i] = Q10[i]/3.0 + 2.0*(Q1[i] + dt*rhsQ1[i])/3.0;
		Q20[i] = Q20[i]/3.0 + 2.0*(Q2[i] + dt*rhsQ2[i])/3.0;
		Q30[i] = Q30[i]/3.0 + 2.0*(Q3[i] + dt*rhsQ3[i])/3.0;
		Q40[i] = Q40[i]/3.0 + 2.0*(Q4[i] + dt*rhsQ4[i])/3.0;
	}
}


__global__
void updateCUDA4(int nElems, int nLoc, double *Q1, double *Q2, double *Q3, double *Q4,
	             double *Q10, double *Q20, double *Q30, double *Q40) {
	int gtid = threadIdx.x + blockIdx.x*blockDim.x;
	for (int i = gtid; i < nElems*nLoc; i += gridDim.x*blockDim.x)
	{
		Q1[i] = Q10[i];
		Q2[i] = Q20[i];
		Q3[i] = Q30[i];
		Q4[i] = Q40[i];
	}
}


__device__
inline void volumeFluxF(int N, double *Q, double *F) {
	for (int i = 0; i < N; ++i)
	{
		double rho, v1, v2, p;

		rho = Q[i];
		v1 = Q[N+i]/rho;
		v2 = Q[2*N+i]/rho;
		p = (1.4-1.0)*(Q[3*N+i] - 0.5*rho*(v1*v1+v2*v2));

		F[i] = Q[N+i];
		F[N+i] = Q[N+i]*v1 + p;
		F[2*N+i] = Q[N+i]*v2;
		F[3*N+i] = v1*(Q[3*N+i] + p);
	}
}


__device__
inline void volumeFluxG(int N, double *Q, double *G) {
	for (int i = 0; i < N; ++i)
	{
		double rho, v1, v2, p;

		rho = Q[i];
		v1 = Q[N+i]/rho;
		v2 = Q[2*N+i]/rho;
		p = (1.4-1.0)*(Q[3*N+i] - 0.5*rho*(v1*v1+v2*v2));

		G[i] = Q[2*N+i];
		G[N+i] = Q[N+i]*v2;
		G[2*N+i] = Q[2*N+i]*v2 + p;
		G[3*N+i] = v2*(Q[3*N+i] + p);
	}
}


/* numericalFluxF computes the numerical flux of F along a given edge given
 * the two-sided values in QLeft and QRight. A local Lax-Friedrichs flux is used
 * here.
 *
 * \param [in] QLeft:  solution values of the left state
 * \param [in] QRight: solution values of the right state
 * \param [out] Fh:    4 x nOrder1D buffer to hold the numerical flux values
 */
__device__
inline void numericalFluxF(int nOrder1D, double *QLeft, double *QRight, double *Fh) {
	double *fLeft = new double[4*nOrder1D];
	double *fRight = new double[4*nOrder1D];
	double *alphaF = new double[4*nOrder1D];

	volumeFluxF(nOrder1D, QLeft, fLeft);
	volumeFluxF(nOrder1D, QRight,fRight);

	// compute the maximum eigenvalue of F'(u) for Lax-Friedrichs flux
	for (int i = 0; i < nOrder1D; ++i)
	{
		double rho, v1, v2, p, c;

		rho = 0.5*(QLeft[i] + QRight[i]);
		v1 = 0.5*(QLeft[nOrder1D+i] + QRight[nOrder1D+i])/rho;
		v2 = 0.5*(QLeft[2*nOrder1D+i] + QRight[2*nOrder1D+i])/rho;
		p = 0.4*(0.5*(QLeft[3*nOrder1D+i] + QRight[3*nOrder1D+i]) - 0.5*rho*(v1*v1+v2*v2));
		c = sqrt(1.4*p/rho);

		alphaF[i] = fabs(v1-c);
		alphaF[nOrder1D+i] = fabs(v1);
		alphaF[2*nOrder1D+i] = fabs(v1);
		alphaF[3*nOrder1D+i] = fabs(v1+c);
	}

	double lambda = 1.0;

	// TODO: should implement a naive max function inline here since
	// std functions will not work in CUDA kernels.
	// double lambda = *std::max_element(alphaF, alphaF+4*nOrder1D);

	for (int i = 0; i < nOrder1D; ++i)
	{
		Fh[i] = 0.5*(fLeft[i] + fRight[i] - lambda*(QLeft[i] - QRight[i]));
		Fh[nOrder1D+i] = 0.5*(fLeft[nOrder1D+i] + fRight[nOrder1D+i] - lambda*(QLeft[nOrder1D+i] - QRight[nOrder1D+i]));
		Fh[2*nOrder1D+i] = 0.5*(fLeft[2*nOrder1D+i] + fRight[2*nOrder1D+i] - lambda*(QLeft[2*nOrder1D+i] - QRight[2*nOrder1D+i]));
		Fh[3*nOrder1D+i] = 0.5*(fLeft[3*nOrder1D+i] + fRight[3*nOrder1D+i] - lambda*(QLeft[3*nOrder1D+i] - QRight[3*nOrder1D+i]));
	}

	delete[] fLeft;
	delete[] fRight;
	delete[] alphaF;
}


/* numericalFluxG computes the numerical flux of G along a given edge given
 * the two-sided values in QLeft and QRight. A local Lax-Friedrichs flux is used
 * here.
 *
 * \param [in] QLeft:  solution values of the left state
 * \param [in] QRight: solution values of the right state
 * \param [out] Gh:    4 x nOrder1D buffer to hold the numerical flux values
 */
__device__
inline void numericalFluxG(int nOrder1D, double *QLeft, double *QRight, double *Gh) {
	double *gLeft = new double[4*nOrder1D];
	double *gRight = new double[4*nOrder1D];
	double *alphaG = new double[4*nOrder1D];

	volumeFluxG(nOrder1D, QLeft, gLeft);
	volumeFluxG(nOrder1D, QRight,gRight);

	// compute maximum eigenvalue of G'(u)
	for (int i = 0; i < nOrder1D; ++i)
	{
		double rho, v1, v2, p, c;

		rho = 0.5*(QLeft[i] + QRight[i]);
		v1 = 0.5*(QLeft[nOrder1D+i] + QRight[nOrder1D+i])/rho;
		v2 = 0.5*(QLeft[2*nOrder1D+i] + QRight[2*nOrder1D+i])/rho;
		p = 0.4*(0.5*(QLeft[3*nOrder1D+i] + QRight[3*nOrder1D+i]) - 0.5*rho*(v1*v1+v2*v2));
		c = sqrt(1.4*p/rho);

		alphaG[i] = fabs(v2-c);
		alphaG[nOrder1D+i] = fabs(v2);
		alphaG[2*nOrder1D+i] = fabs(v2);
		alphaG[3*nOrder1D+i] = fabs(v2+c);
	}

	double lambda = 1.0;
	// TODO: should implement a naive max function inline here since
	// std functions will not work in CUDA kernels.
	// double lambda = *std::max_element(alphaG, alphaG+4*nOrder1D);

	for (int i = 0; i < nOrder1D; ++i)
	{
		Gh[i] = 0.5*(gLeft[i] + gRight[i] - lambda*(QLeft[i] - QRight[i]));
		Gh[nOrder1D+i] = 0.5*(gLeft[nOrder1D+i] + gRight[nOrder1D+i] - lambda*(QLeft[nOrder1D+i] - QRight[nOrder1D+i]));
		Gh[2*nOrder1D+i] = 0.5*(gLeft[2*nOrder1D+i] + gRight[2*nOrder1D+i] - lambda*(QLeft[2*nOrder1D+i] - QRight[2*nOrder1D+i]));
		Gh[3*nOrder1D+i] = 0.5*(gLeft[3*nOrder1D+i] + gRight[3*nOrder1D+i] - lambda*(QLeft[3*nOrder1D+i] - QRight[3*nOrder1D+i]));
	}

	delete[] gLeft;
	delete[] gRight;
	delete[] alphaG;
}
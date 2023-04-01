#ifndef MESHING_HPP
#define MESHING_HPP

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

/* mesh is a class containing the mesh coordinates and connectivity as well as
 * the element neighbor connections for the 2D Euler equations with natural boundary
 * conditions. The default constructor takes the following arguments:
 *
 * \param [in] xleft:  left boundary of domain
 * \param [in] xright: right boundary of domain
 * \param [in] yleft:  bottom boundary of domain
 * \param [in] yright: top boundary of domain
 * \param [in] dimx:   number of mesh elements in x direction
 * \param [in] dimy:   number of mesh elements in y direction
 *
 */
class mesh {
  private:
    int Nx, Ny, numElem, numVert;
    double x0, xN, y0, yN, hx, hy;

    int *EToV, *mapB;
    double *Vxy;

  public: 
  	mesh(double xleft, double xright, double yleft, double yright, int dimx, int dimy) {
  	  // set mesh spacing
  	  x0 = xleft;	xN = xright;
  	  y0 = yleft;	yN = yright;

      Nx = dimx;
      Ny = dimy;
      hx = (xN-x0) / Nx;
      hy = (yN-y0) / Ny;

      // set number of cells and vertices
      numElem = Nx*Ny;
      numVert = (Nx+1)*(Ny+1);

      // EToV stores connectivity; first four entries of each row give
      // the vertices for that element; last two entries give the i and j
      // indices of the cell on the grid of cells
      //
      // mapB stores the four neighbors of each cell in the order left,
      // right, bottom, top
      // EToV = new int[numElem * 6]();
      // mapB = new int[numElem * 4];
      // Vxy = new double[numVert * 2];

      cudaMallocManaged(&EToV, numElem*6*sizeof(int));
      cudaMallocManaged(&mapB, numElem*4*sizeof(int));
      cudaMallocManaged(&Vxy, numVert*2*sizeof(double));
  	}

  	// default destructor
  	~mesh() {
  	  // delete[] EToV;
  	  // delete[] mapB;
  	  // delete[] Vxy;

  	  cudaFree(EToV);
  	  cudaFree(mapB);
  	  cudaFree(Vxy);
  	}

    void quadConnectivity2D();

    int getNumCells();

    int getNumVerts();


    // access functions for mesh arrays; these are only necessary
    // if one wants to access these arrays directly (not recommended or necessary);
    // note that DGsolver is a friend class to mesh, so the arrays EToV, Vxy, and
    // mapB may be accessed directly within DGsolver.
  	const int* getEToV() const {
  	  return EToV;
  	}

  	const int* getmapB() const {
  	  return mapB;
  	}

  	const double* getVxy() const {
  	  return Vxy;
  	}


    friend class DGsolver;
};


/* getNumCells returns the number of elements in the mesh.
 */
inline int mesh::getNumCells() {
  return numElem;
}


/* getNumVerts returns the number of nodes in the mesh.
 */
inline int mesh::getNumVerts() {
  return numVert;
}


/* quadConnectivity2D initializes the connectivity and boundary arrays for the
 * mesh. Connecivity is always assumed counterclockwise so that the Jacobian
 * of the mapping from the reference element to physical element has positive sign.
 * For the boundary, we assume natural BCs, i.e. the normal derivative is zero along 
 * the boundary. The BCs are enforced weakly in the weak formulation of the scheme.
 * In particular, they are enforced when computing the numerical flux along boundary
 * edges. For cells on the left boundary, we set its left neighbor to be itself, which
 * is a crude way to approximate zero derivative in the normal direction.
 */
inline void mesh::quadConnectivity2D() {

	// vertices and connecivity
	for (int i = 0; i < Ny+1; ++i)
	{
		for (int j = 0; j < Nx+1; ++j)
		{
			int idx = i*(Nx+1)+j;

			Vxy[idx*2] = x0 + j*hx;
			Vxy[idx*2+1] = y0 + i*hy;

			if ((i < Ny) && (j < Nx))
			{
				int v1 = i*(Nx+1) + j;
				int v2 = i*(Nx+1) + j+1;
				int v3 = (i+1)*(Nx+1) + j+1;
				int v4 = (i+1)*(Nx+1) + j;

				int jdx = i*Nx+j;
				EToV[jdx*6] = v1;
				EToV[jdx*6+1] = v2;
				EToV[jdx*6+2] = v3;
				EToV[jdx*6+3] = v4;
				EToV[jdx*6+4] = i;
				EToV[jdx*6+5] = j;
			}
		}
	}

	// neighbors of each cell
	for (int i = 0; i < Ny; ++i)
	{
		for (int j = 0; j < Nx; ++j)
		{
			int idx = i*Nx + j;

			mapB[idx*4] = i*Nx + ((j+Nx-1)%Nx);
			mapB[idx*4+1] = i*Nx + ((j+1)%Nx);
			mapB[idx*4+2] = ((i+Ny-1)%Ny)*Nx + j;
			mapB[idx*4+3] = ((i+1)%Ny)*Nx + j;

			// // elements on left boundary have left neighbors equal to themselves
			// if (j == 0)
			// {
			// 	mapB[idx*4] = i*Nx + j;
			// }

			// // elements on right boundary have right neighbors equal to themselves
			// if (j == Nx-1)
			// {
			// 	mapB[idx*4+1] = i*Nx + j;
			// }

			// elements on bottom boundary have bottom neighbors equal to themselves
			if (i == 0)
			{
				mapB[idx*4+2] = i*Nx + j;
			}

			// elements on the top boundary have top neighbors equal to themselves
			if (i == Ny-1)
			{
				mapB[idx*4+3] = i*Nx + j;
			}
		}
	}
}

#endif
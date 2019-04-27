#include <cmath>

/* meshParameters generates the domain and mesh resolution parameters for
 * the DG method. This currently only implements rectangular domains.
 *
 * \param [out] x0: left boundary of rectangular domain
 * \param [out] xN: right boundary of the domain
 * \param [out] y0: bottom boundary of the domain
 * \param [out] yN: top boundary of the domain
 * \param [out] Nx: number of elements in the x direction
 * \param [out] Ny: number of elements in the y direction
 */
void meshParameters(double &x0, double &xN, double &y0, double &yN, int &Nx, int &Ny) {
	x0 = 0.0;
	xN = 0.25;
	y0 = 0.0;
	yN = 1.0;

	Nx = 64;
	Ny = 256;
}


/* dgParameters generate the time-stepping and polynomial basis parameters 
 * for the DG method. This currently only supports uniform time-stepping,
 * although CFL condition based on maximum eigenvalues of the Jacobian should
 * be implemented soon.
 *
 * \param [out] pdeg: degree of the tensor product Legendre basis functions; t
 *					  there are (pdeg+1)^2 basis functions total
 * \param [out] T:    stopping time of the simulation
 * \param [out] dt:   time step for the Runge-Kutta time stepper
 */
void dgParameters(int &pdeg, double &T, double &dt) {
	pdeg = 2;
	T = 0.4;
	dt = 0.005;
}


/* initialCondition evaluates the initial condition of the system at the 
 * given point (x,y). Currently, this routine cannot take in an array of
 * points. Low priority to add this feature since the routine is used only
 * once to project the initial data into the basis space.
 *
 * \param [in] x:  x coordinate at which to evaluate initial condition
 * \param [in] y:  y cooridnate at which to evaluate initial condition
 * \param [out] q: buffer in which initial data for the conserved variables
 *				   is held: (q0,q1,q2,q3) = (density, x momentum, y momentum,
 *				   energy).
 */
void initialCondition(double x, double y, double *q) {
    double rho, v1, v2, p;

	if (y >= 0.5)
	{
		rho = 1.0;
		p = y+1.5;
	}
	else {
		rho = 2.0;
		p = 2.0*y+1.0;
	}

	double c = sqrt(1.4*p/rho);
	v1 = 0.0;
	v2 = -0.025*c*cos(8.0*M_PI*x);

	q[0] = rho;
	q[1] = rho*v1;
	q[2] = rho*v2;
	q[3] = p/(1.4-1.0) + 0.5*rho*(v1*v1 + v2*v2);
}

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cmath>

void meshParameters(double &x0, double &xN, double &y0, double &yN, int &Nx, int &Ny);

void dgParameters(int &pdeg, double &T, double &dt);

void initialCondition(double x, double y, double *q);

#endif
clear all; close all; clc;

N = [128];

% computational domain
x0 = 0;
xN = 1;
y0 = 0;
yN = 1;

% degree of basis functions
p = 1;

% stopping time
T = 0.25;

% buffer to store L2 error
L2 = zeros(length(N),1);
rateL2 = zeros(length(N)-1,1);

for i=1:length(N)
    % set discretization parameters
    Nx = N(i);
    Ny = N(i);
    
    dx = (xN-x0)/Nx;
    dy = (yN-y0)/Ny;

    % generate mesh
    [VX,VY,EToV,mapB] = meshGenQuad(Nx,Ny,x0,xN,y0,yN);  

    % compute L2 projection of initial condition into basis space
    Q0 = projL2(p,VX,VY,EToV);

    % integrate over time
    Q = timeStepper(p,dx,dy,Nx,Ny,VX,VY,EToV,mapB,Q0,T);

    % plot solution at cell vertices
    plotSolution(p,x0,xN,y0,yN,VX,VY,EToV,Q);
    
    % compute error in L2 functional norm
    L2(i) = computeError(p,dx,dy,VX,VY,EToV,Q,T);
end


% compute L2 convergence rates
for i=1:length(N)-1
    rateL2(i) = log2(L2(i)/L2(i+1));
end


fprintf('Convergence Results for rho(x,y,t) = 1+0.25*sin(x+y-2t)\n');
% fprintf('L2 Error    |      L2 rate\n');
[L2 [NaN; rateL2]]

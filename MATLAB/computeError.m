function L2 = computeError(p,dx,dy,VX,VY,EToV,Q,T)

Nloc = (p+1)^2;
K = size(EToV,1);

L2 = 0.0;

% load quadrature rule
[w,x,y] = quadRule2D(p+2);

% compute basis functions at quadrature nodes
[phi,dphix,dphiy] = basisFunctions(p,x',y',dx,dy);

for k=1:K
    xa = VX(EToV(k,1));   xb = VX(EToV(k,2));
    ya = VY(EToV(k,1));   yb = VY(EToV(k,3));
    
    % determinant of transformation Jacobian from [-1,1]^2 to element K
    vol = (xb-xa)*(yb-ya)/4;
    
    % compute local solution
    c = Q(k,:,1);
    uloc = c*phi;
    
    % map quadrature nodes to element K
    xk = (xb-xa)/2.0*x + (xb+xa)/2.0;
    yk = (yb-ya)/2.0*y + (yb+ya)/2.0;
    
    ue = exactSolution(xk',yk',T);
    
    L2 = L2 + vol*sum(w'.*(uloc-ue).^2);
end

L2 = sqrt(L2);
    
return
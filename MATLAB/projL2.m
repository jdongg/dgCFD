function Q0 = projL2(p,VX,VY,EToV)

% number of elements
K = size(EToV,1);

% local problem size
Nloc = (p+1)^2;
Q0 = zeros(K,Nloc,4);

f = zeros(Nloc,4);

% load quadrature rule
[w,x,y] = quadRule2D(p+2);

% compute basis functions at quadrature nodes
phi = basisFunctions(p,x',y');

% load initial condition
IC = @initialCondition;

for k=1:K
    xa = VX(EToV(k,1));   xb = VX(EToV(k,2));
    ya = VY(EToV(k,1));   yb = VY(EToV(k,3));
    
    % map quadrature nodes to element K
    xk = (xb-xa)/2.0*x + (xb+xa)/2.0;
    yk = (yb-ya)/2.0*y + (yb+ya)/2.0;
    
    % compute local right-hand side; not sure how to vectorize this
    for i=1:Nloc
        f(i,:) = sum(w'.*IC(xk,yk)'.*phi(i,:),2)';
    end
 
    % linear solve to obtain basis function coefficients; note that the
    % mass matrix on cell k is just the identity matrix scaled by the
    % volume, i.e. Mc = f is equivalent to Ic = f/vol, or c = f/vol.
    c = f;
    Q0(k,:,:) = c;
end

return
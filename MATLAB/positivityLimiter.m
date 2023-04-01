function u = positivityLimiter(p,m,M,VX,VY,EToV,u)

K = size(EToV,1);

% load quadrature rule
[w,x,y] = quadRule2D(p+3);

% compute basis functions at quadrature nodes
phi = basisFunctions(p,x',y');

% compute basis functions at vertices of cell
xloc = [-1 1 1 -1];
yloc = [-1 -1 1 1];
philoc = basisFunctions(p,xloc,yloc);

% % global max and min values
% M = 1.0;
% m = 0.0;

for k=1:K    
    xa = VX(EToV(k,1));   xb = VX(EToV(k,2));
    ya = VY(EToV(k,1));   yb = VY(EToV(k,3));
    
    % determinant of transformation Jacobian from [-1,1]^2 to element K
    vol = (xb-xa)*(yb-ya)/4;
    
    % compute local solution
    c = u(k,:);
    uloc = c*phi;
    
    % compute cell average
    avg = vol*sum(w'.*uloc)/((xb-xa)*(yb-ya));
    
    uloc = [uloc, c*philoc];
    
    % max and min on cell k
    Mk = max(uloc);
    mk = min(uloc);
    
    % scaling parameter for MPP limiter
    theta = min([abs((M-avg)/(Mk-avg)), abs((m-avg)/(mk-avg)), 1.0]);
    
    % modify the polynomial as follows:
    %
    %   p_mod(x) = theta*(p(x) - u_avg) + u_avg
    %
    % note that all coefficients are simply scaled by theta while the first
    % coefficient (corresponding to the constant basis function) is
    % translated by -2*u_avg*(theta-1) because the constant 
    cmod = theta*c;
    cmod(1) = cmod(1) - 2.0*avg*(theta-1.0);
    
    u(k,:) = cmod;
end

return
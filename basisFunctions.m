function [phi,dphix,dphiy] = basisFunctions(p,x,y,dx,dy)

if (nargin < 5)
    dx = 1.0;
    dy = 1.0;
end

% compute tensor product Legendre basis functions at the points (x,y)
Nloc = (p+1)^2;
phi = zeros(Nloc,length(x));
dphix = zeros(Nloc,length(x));
dphiy = zeros(Nloc,length(x));

phiTmpX = zeros(p+1,length(x));
phiTmpY = zeros(p+1,length(y));
dphiTmpX = zeros(p+1,length(x));
dphiTmpY = zeros(p+1,length(y));

switch(p)
    case 0
        phi(1,:) = 0.5*ones(1,length(x));
        dphix(1,:) = zeros(1,length(x));
        dphiy(1,:) = zeros(1,length(x));
    case 1
        phiTmpX(1,:) = ones(1,length(x));
        phiTmpX(2,:) = x;
        dphiTmpX(1,:) = zeros(1,length(x));
        dphiTmpX(2,:) = ones(1,length(x));
        
        phiTmpY(1,:) = ones(1,length(y));
        phiTmpY(2,:) = y;
        dphiTmpY(1,:) = zeros(1,length(y));
        dphiTmpY(2,:) = ones(1,length(y));
    case 2
        phiTmpX(1,:) = ones(1,length(x));
        phiTmpX(2,:) = x;
        phiTmpX(3,:) = 0.5*(3.0*x.^2 - 1.0);
        dphiTmpX(1,:) = zeros(1,length(x));
        dphiTmpX(2,:) = ones(1,length(x));
        dphiTmpX(3,:) = 3.0*x;
        
        phiTmpY(1,:) = ones(1,length(y));
        phiTmpY(2,:) = y;
        phiTmpY(3,:) = 0.5*(3.0*y.^2 - 1.0);
        dphiTmpY(1,:) = zeros(1,length(y));
        dphiTmpY(2,:) = ones(1,length(y));
        dphiTmpY(3,:) = 3.0*y;
    case 3
        phiTmpX(1,:) = ones(1,length(x));
        phiTmpX(2,:) = x;
        phiTmpX(3,:) = 0.5*(3.0*x.^2 - 1.0);
        phiTmpX(4,:) = 0.5*(5.0*x.^3 - 3.0*x);
        dphiTmpX(1,:) = zeros(1,length(x));
        dphiTmpX(2,:) = ones(1,length(x));
        dphiTmpX(3,:) = 3.0*x;
        dphiTmpX(4,:) = 15.0*x.^2/2.0 - 1.5*x;
        
        phiTmpY(1,:) = ones(1,length(y));
        phiTmpY(2,:) = y;
        phiTmpY(3,:) = 0.5*(3.0*y.^2 - 1.0);
        phiTmpY(4,:) = 0.5*(5.0*y.^3 - 3.0*y);
        dphiTmpY(1,:) = zeros(1,length(y));
        dphiTmpY(2,:) = ones(1,length(y));
        dphiTmpY(3,:) = 3.0*y;
        dphiTmpY(4,:) = 15.0*y.^2/2.0 - 1.5*y;
    case 4
        phiTmpX(1,:) = ones(1,length(x));
        phiTmpX(2,:) = x;
        phiTmpX(3,:) = 0.5*(3.0*x.^2 - 1.0);
        phiTmpX(4,:) = 0.5*(5.0*x.^3 - 3.0*x);
        phiTmpX(5,:) = (35*x.^4 - 30*x.^2 + 3.0)/8.0;
        dphiTmpX(1,:) = zeros(1,length(x));
        dphiTmpX(2,:) = ones(1,length(x));
        dphiTmpX(3,:) = 3.0*x;
        dphiTmpX(4,:) = 15.0*x.^2/2.0 - 1.5*x;
        dphiTmpX(5,:) = (140*x.^3 - 60*x.^2)/8.0;
        
        phiTmpY(1,:) = ones(1,length(y));
        phiTmpY(2,:) = y;
        phiTmpY(3,:) = 0.5*(3.0*y.^2 - 1.0);
        phiTmpY(4,:) = 0.5*(5.0*y.^3 - 3.0*y);
        phiTmpY(5,:) = (35*y.^4 - 30*y.^2 + 3.0)/8.0;
        dphiTmpY(1,:) = zeros(1,length(y));
        dphiTmpY(2,:) = ones(1,length(y));
        dphiTmpY(3,:) = 3.0*y;
        dphiTmpY(4,:) = 15.0*y.^2/2.0 - 1.5*y;
        dphiTmpY(5,:) = (140*y.^3 - 60*y.^2)/8.0;
end

if (p>0)
    for i=1:p+1
        for j=1:p+1
            c = sqrt((2*i-1)*(2*j-1))/2.0;
            phi((i-1)*(p+1)+j,:) = c*phiTmpX(i,:).*phiTmpY(j,:);
            dphix((i-1)*(p+1)+j,:) = c*dphiTmpX(i,:)/(dx/2).*phiTmpY(j,:);
            dphiy((i-1)*(p+1)+j,:) = c*phiTmpX(i,:).*dphiTmpY(j,:)/(dy/2);
        end
    end
end

return
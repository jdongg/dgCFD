function G = volumeFluxG(Q)

gamma = 1.4;

G(:,1) = Q(:,3);

rho = Q(:,1);
v1 = Q(:,2)./rho;
v2 = Q(:,3)./rho;
p = (gamma-1.0)*(Q(:,4) - 0.5*rho.*(v1.^2+v2.^2));

G(:,2) = Q(:,2).*v2 ;
G(:,3) = Q(:,3).*v2 + p;
G(:,4) = v2.*(Q(:,4)+p);

return
function F = volumeFluxF(Q)

gamma = 1.4;

F(:,1) = Q(:,2);

rho = Q(:,1);
v1 = Q(:,2)./rho;
v2 = Q(:,3)./rho;
p = (gamma-1.0)*(Q(:,4) - 0.5*rho.*(v1.^2+v2.^2));

F(:,2) = Q(:,2).*v1 + p;
F(:,3) = Q(:,2).*v2;
F(:,4) = v1.*(Q(:,4)+p);

return
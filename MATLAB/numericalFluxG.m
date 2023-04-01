function Gh = numericalFluxG(QLeft,QRight)

gamma = 1.4;

% using the local Lax-Friedrichs flux
gLeft = volumeFluxG(QLeft);
gRight = volumeFluxG(QRight);

% alpha should be chosen according to alpha = max |g'(u)|
% rho = QLeft(:,1);
% v1 = QLeft(:,2)./rho;
% v2 = QLeft(:,3)./rho;
% p = (gamma-1.0)*(QLeft(:,4) - 0.5*rho.*(v1.^2+v2.^2));
% c = sqrt(gamma*p./rho);
% 
% alphaGL = abs([v2-c v2 v2 v2+c]);
% 
% rho = QRight(:,1);
% v1 = QRight(:,2)./rho;
% v2 = QRight(:,3)./rho;
% p = (gamma-1.0)*(QRight(:,4) - 0.5*rho.*(v1.^2+v2.^2));
% c = sqrt(gamma*p./rho);
% 
% alphaGR = abs([v2-c v2 v2 v2+c]);
% alphaG = max(max(max(alphaGL,alphaGR)));

Qavg = 0.5*(QLeft+QRight);
rho = Qavg(:,1);
v1 = Qavg(:,2)./rho;
v2 = Qavg(:,3)./rho;
p = (gamma-1.0)*(Qavg(:,4) - 0.5*rho.*(v1.^2+v2.^2));
c = sqrt(gamma*p./rho);

alphaG = max(max(abs([v2-c v2 v2 v2+c])));

% alphaG = 1.0;

Gh = 0.5*(gLeft+gRight - alphaG.*(QLeft-QRight));

return
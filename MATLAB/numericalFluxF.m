function Fh = numericalFluxF(QLeft,QRight)

gamma = 1.4;

% using the local Lax-Friedrichs flux
fLeft = volumeFluxF(QLeft);
fRight = volumeFluxF(QRight);

% alpha should be chosen according to alpha = max |f'(u)|
% rho = QLeft(:,1);
% v1 = QLeft(:,2)./rho;
% v2 = QLeft(:,3)./rho;
% p = (gamma-1.0)*(QLeft(:,4) - 0.5*rho.*(v1.^2+v2.^2));
% c = sqrt(gamma*p./rho);
% 
% alphaFL = abs([v1-c v2 v2 v1+c]);
% 
% rho = QRight(:,1);
% v1 = QRight(:,2)./rho;
% v2 = QRight(:,3)./rho;
% p = (gamma-1.0)*(QRight(:,4) - 0.5*rho.*(v1.^2+v2.^2));
% c = sqrt(gamma*p./rho);
% 
% alphaFR = abs([v1-c v1 v1 v1+c]);
% alphaF = max(max(max(alphaFL,alphaFR)));


Qavg = 0.5*(QLeft+QRight);
rho = Qavg(:,1);
v1 = Qavg(:,2)./rho;
v2 = Qavg(:,3)./rho;
p = (gamma-1.0)*(Qavg(:,4) - 0.5*rho.*(v1.^2+v2.^2));
c = sqrt(gamma*p./rho);

alphaF = max(max(abs([v1-c v1 v1 v1+c])));

% alphaF = 1.0;

Fh = 0.5*(fLeft+fRight - alphaF.*(QLeft-QRight));

return
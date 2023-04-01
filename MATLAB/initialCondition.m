function Q0 = initialCondition(x,y)

Q0 = zeros(length(x),4);

gamma = 1.4;

% Q0(:,1) = 1.0 + 0.25*sin(x+y);

rho = zeros(length(x),1);
v1 = zeros(length(x),1);
v2 = zeros(length(x),1);
p = zeros(length(x),1);

for i=1:length(x)
%     % first Riemann test case
%     if (x(i) <= 0.5 && y(i) >= 0.5)
%         rho(i) = 1.0;
%         p(i) = 1.0;
%         v1(i) = 0.7276*rho(i);
%         v2(i) = 0.0*rho(i);
%     elseif (x(i) > 0.5 && y(i) >= 0.5) 
%         rho(i) = 0.5313;
%         p(i) = 0.4;
%         v1(i) = 0.0*rho(i);
%         v2(i) = 0.0*rho(i);
%     elseif (x(i) <= 0.5 && y(i) < 0.5)
%         rho(i) = 0.8;
%         p(i) = 1.0;
%         v1(i) = 0.0*rho(i);
%         v2(i) = 0.0*rho(i);
%     else 
%         rho(i) = 1.0;
%         p(i) = 1.0;
%         v1(i) = 0.0*rho(i);
%         v2(i) = 0.7276*rho(i);
%     end
    
    
    % Rayleigh Taylor instability
    if (y(i) <= 0.5)
        rho(i) = 2.0;
        p(i) = 2.0*y(i) + 1.0;
    else
        rho(i) = 1.0;
        p(i) = y(i) + 1.5;
    end

    c = sqrt(1.4*p(i)/rho(i));
    v1(i) = 0.0;
    v2(i) = -0.025*c*cos(8.0*pi*x(i));
end

Q0(:,1) = rho;
Q0(:,2) = Q0(:,1).*v1;
Q0(:,3) = Q0(:,1).*v2;
Q0(:,4) = p/(gamma-1.0) + 0.5*Q0(:,1).*(v1.^2+v2.^2);

return
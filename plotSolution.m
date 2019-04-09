function plotSolution(p,x0,xN,y0,yN,Vx,Vy,EToV,Q)

K = size(EToV,1);

% xp = zeros(K,4);
% yp = zeros(K,4);
% zp = zeros(K,4);
% quadm = zeros(K,4);

figure
for k=1:K
    v1 = EToV(k,1);
    v2 = EToV(k,2);
    v3 = EToV(k,3);
    v4 = EToV(k,4);
    
    Vloc = [Vx(v1) Vy(v1);
            Vx(v2) Vy(v2);
            Vx(v3) Vy(v3);
            Vx(v4) Vy(v4)];
        
    c = Q(k,:,1);
    
    x = [-1; 1; 1; -1];
    y = [-1; -1; 1; 1];
    phi = basisFunctions(p,x',y');
    
    uloc = c*phi;
    
%     xp(k,:) = Vloc(:,1)';
%     yp(k,:) = Vloc(:,2)';
%     zp(k,:) = uloc;
%     quadm(k,:) = [(k-1)*4+1 (k-1)*4+2 (k-1)*4+3 (k-1)*4+4];
    
    patch(Vloc(:,1),Vloc(:,2),uloc',uloc');
    colormap('jet');
    xlim([x0 xN]);
    ylim([y0 yN]);
    hold on;
end

% pbaspect([1 4 1]);
% quadmesh(quadm,xp,yp,zp);
% shading interp;
return
function plotMesh(Vx,Vy,EToV)

figure
for i=1:size(EToV,1)
    v1 = EToV(i,1);
    v2 = EToV(i,2);
    v3 = EToV(i,3);
    v4 = EToV(i,4);
    
    Vloc = [Vx(v1) Vy(v1);
            Vx(v2) Vy(v2);
            Vx(v3) Vy(v3);
            Vx(v4) Vy(v4)];
    line(Vloc(:,1),Vloc(:,2));
    hold on;
end

return
function [Vx,Vy,EToV,mapB] = meshGenQuad(Nx,Ny,x0,xN,y0,yN)

Vx = zeros((Nx+1)*(Ny+1),1);
Vy = zeros((Nx+1)*(Ny+1),1);
EToV = zeros(Nx*Ny,6);

mapB.Left = zeros(Ny,Nx);
mapB.Right = zeros(Ny,Nx);
mapB.Down = zeros(Ny,Nx);
mapB.Up = zeros(Ny,Nx);

hx = (xN-x0)/Nx;
hy = (yN-y0)/Ny;

for i=1:Ny+1
    for j=1:Nx+1
        Vx((i-1)*(Nx+1)+j) = x0 + (j-1)*hx;
        Vy((i-1)*(Nx+1)+j) = y0 + (i-1)*hy;
        
        if ((i<Ny+1) && (j<Nx+1))
            v1 = (i-1)*(Nx+1) + j;
            v2 = (i-1)*(Nx+1) + j+1;
            v3 = i*(Nx+1) + j+1;
            v4 = i*(Nx+1) + j;
            EToV((i-1)*Nx+j,:) = [v1 v2 v3 v4 i j];
        end
    end
end

for i=1:Ny
    for j=1:Nx
        % interior neighbors
        mapB.Left(i,j) = (i-1)*Nx + mod(j+Nx-2,Nx)+1;
        mapB.Right(i,j) = (i-1)*Nx + mod(j,Nx)+1;
        mapB.Down(i,j) = mod(i+Ny-2,Ny)*Nx + j;
        mapB.Up(i,j) = (mod(i,Ny))*Nx + j;
        
        if (j==1)
            % cells on left boundary
            mapB.Left(i,j) = (i-1)*Nx + j;
        end
        
        if (j==Nx)
            % cells on right boundary
            mapB.Right(i,j) = (i-1)*Nx + j;
        end
        
        if (i==1)
            % cells on bottom boundary
            mapB.Down(i,j) = (i-1)*Nx + j;
        end
        
        if (i==Ny)
            % cells on top boundary
            mapB.Up(i,j) = (i-1)*Nx + j;
        end
    end
end

return
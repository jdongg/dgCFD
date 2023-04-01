function Q = timeStepper(p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB,Q0,T)

disp('WARNING: If positivity-preserving limiter is being used, you must manually input the values of M and m');
M = [1.25 1.25 1.25 1.25+1/0.4];
m = [0.75 0.75 0.75 0.75+1/0.4];

t = 0.0;
dt = 0.001;
while (t<T)
    % third-order TVD time-stepping
    rhsQ = computeRHS(Q0,p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB);
    Q = Q0 + dt*rhsQ;
    
    % apply the moment limiter and positivity limiter component-wise
    for j=1:4
        Q(:,:,j) = momentLimiter(p,Nx,Ny,EToV,mapB,Q(:,:,j));
%         Q(:,:,j) = positivityLimiter(p,m(j),M(j),Vx,Vy,EToV,Q(:,:,j));
    end
    
    rhsQ = computeRHS(Q,p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB);
    Q = 3.0/4.0*Q0 + 1.0/4.0*(Q + dt*rhsQ);
    for j=1:4
        Q(:,:,j) = momentLimiter(p,Nx,Ny,EToV,mapB,Q(:,:,j));
%         Q(:,:,j) = positivityLimiter(p,m(j),M(j),Vx,Vy,EToV,Q(:,:,j));
    end
    
    rhsQ = computeRHS(Q,p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB);
    Q0 = Q0/3.0 + 2.0/3.0*(Q + dt*rhsQ);
    for j=1:4
        Q0(:,:,j) = momentLimiter(p,Nx,Ny,EToV,mapB,Q0(:,:,j));
%         Q0(:,:,j) = positivityLimiter(p,m(j),M(j),Vx,Vy,EToV,Q0(:,:,j));
    end
  
    t = t+dt
end

% last time step to reach T
dt = T-t;
rhsQ = computeRHS(Q0,p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB);
Q = Q0 + dt*rhsQ;
for j=1:4
    Q(:,:,j) = momentLimiter(p,Nx,Ny,EToV,mapB,Q(:,:,j));
%     Q(:,:,j) = positivityLimiter(p,m(j),M(j),Vx,Vy,EToV,Q(:,:,j));
end

rhsQ = computeRHS(Q,p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB);
Q = 3.0/4.0*Q0 + 1.0/4.0*(Q + dt*rhsQ);
for j=1:4
    Q(:,:,j) = momentLimiter(p,Nx,Ny,EToV,mapB,Q(:,:,j));
%     Q(:,:,j) = positivityLimiter(p,m(j),M(j),Vx,Vy,EToV,Q(:,:,j));
end

rhsQ = computeRHS(Q,p,dx,dy,Nx,Ny,Vx,Vy,EToV,mapB);
Q = Q0/3.0 + 2.0/3.0*(Q + dt*rhsQ);
for j=1:4
    Q(:,:,j) = momentLimiter(p,Nx,Ny,EToV,mapB,Q(:,:,j));
%     Q(:,:,j) = positivityLimiter(p,m(j),M(j),Vx,Vy,EToV,Q(:,:,j));
end

return
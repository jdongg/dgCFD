function rhsQ = computeRHS(Q0,p,dx,dy,Nx,Ny,VX,VY,EToV,mapB)

Nloc = (p+1)^2;
K = size(EToV,1);

rhsQ = zeros(K,Nloc,4);

% load quadrature rule
[w,x,y] = quadRule2D(p+2);
w1d = quadRule1D(p+1);

% compute basis functions at quadrature nodes
[phiV,dphixV,dphiyV] = basisFunctions(p,x',y',dx,dy);

% compute basis functions on all edges
phiEdge = basisFunctionsEdge(p);

for k=1:K
    xa = VX(EToV(k,1));   xb = VX(EToV(k,2));
    ya = VY(EToV(k,1));   yb = VY(EToV(k,3));
    
    % determinant of transformation Jacobian from [-1,1]^2 to element K
    vol = (xb-xa)*(yb-ya)/4;
    
    % compute local solution
%     c = squeeze(Q0(k,:,:));
    c = reshape(Q0(k,:,:),[Nloc 4]);
    Q = c'*phiV; % each column of Q corresponds to a conserved variable
    
    % each column of F and G correspond to a conserved variable
    F = volumeFluxF(Q');
    G = volumeFluxG(Q');

    % compute volume contributions  
    for i=1:Nloc
        rhsQ(k,i,:) = vol*sum(w'.*(dphixV(i,:).*F' + dphiyV(i,:).*G'),2);
        
%         % gravity terms for RT instability
%         rhsQ(k,i,3) = rhsQ(k,i,3) + vol*sum(w'.*Q(1,:).*phiV(i,:),2);
%         rhsQ(k,i,4) = rhsQ(k,i,4) + vol*sum(w'.*Q(3,:).*phiV(i,:),2);
    end
    
    
    
    % compute surface contributions
    idx = EToV(k,5);
    jdx = EToV(k,6);
    
    % lookup neighbor cells to cell (i,j), assuming periodic boundaries
%     nLeft = (idx-1)*Ny + mod(jdx+Nx-2,Nx)+1;
%     nRight = (idx-1)*Ny + mod(jdx,Nx)+1;
%     nDown = mod(idx+Ny-2,Ny)*Ny + jdx;
%     nUp = (mod(idx,Ny))*Ny + jdx;
    
    nLeft = mapB.Left(idx,jdx);
    nRight = mapB.Right(idx,jdx);
    nDown = mapB.Down(idx,jdx);
    nUp = mapB.Up(idx,jdx);

    % pull basis coefficients on neighbor cells
    cL = reshape(Q0(nLeft,:,:),[Nloc 4]);
    cR = reshape(Q0(nRight,:,:),[Nloc 4]);
    cD = reshape(Q0(nDown,:,:),[Nloc 4]);
    cU = reshape(Q0(nUp,:,:),[Nloc 4]);
    
    
    %% left edge
    Q = c'*phiEdge.Left;
    if (jdx>1)
        QLeft = cL'*phiEdge.Right;
    else
        QLeft = c'*phiEdge.Left;
    end
    
    % determinant of Jacobian for 1D transformation
    vol1d = (xb-xa)/2;
    
    Fh = numericalFluxF(Q',QLeft');
    for i=1:Nloc
        rhsQ(k,i,:) = rhsQ(k,i,:) + reshape(vol1d*sum(w1d'.*phiEdge.Left(i,:).*Fh',2)',[1 1 4]);
    end
    
    
    %% bottom edge
    Q = c'*phiEdge.Down;
    if (idx>1)
        QDown = cD'*phiEdge.Up;
    else
        QDown = c'*phiEdge.Down;
    end
    
    Gh = numericalFluxG(Q',QDown'); 
    for i=1:Nloc
        rhsQ(k,i,:) = rhsQ(k,i,:) + reshape(vol1d*sum(w1d'.*phiEdge.Down(i,:).*Gh',2)',[1 1 4]);
    end
    
    %% right edge
    Q = c'*phiEdge.Right;  
    if (jdx<Nx)
        QRight = cR'*phiEdge.Left;
    else
        QRight = c'*phiEdge.Right;
    end
    
    Fh = numericalFluxF(QRight',Q');
    for i=1:Nloc
        rhsQ(k,i,:) = rhsQ(k,i,:) - reshape(vol1d*sum(w1d'.*phiEdge.Right(i,:).*Fh',2)',[1 1 4]);
    end
    
    
    %% top edge
    Q = c'*phiEdge.Up;
    if (idx<Ny)
        QUp = cU'*phiEdge.Down;
    else
        QUp = c'*phiEdge.Up;
    end
    
    Gh = numericalFluxG(QUp',Q');
    for i=1:Nloc
        rhsQ(k,i,:) = rhsQ(k,i,:) - reshape(vol1d*sum(w1d'.*phiEdge.Up(i,:).*Gh',2)',[1 1 4]);
    end
    
    % we have Lu = inv(M)*rhsU for an explicit time-stepping scheme, but
    % the Legendre basis is orthonormal, so on each element, the mass
    % matrix is the identity matrix scaled by the a fourth of the cell area
    rhsQ(k,:,:) = rhsQ(k,:,:)/vol;
    
end

return
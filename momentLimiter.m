function u = momentLimiter(p,Nx,Ny,EToV,mapB,u)

K = size(EToV,1);

tol = 1.e-12;

for k=1:K
    % pull coefficients from current element
    c = u(k,:);
    
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
    cL = u(nLeft,:);
    cR = u(nRight,:);
    cD = u(nDown,:);
    cU = u(nUp,:);
    
    % coefficients for the minmod limiter. larger values will give more
    % dissipative results, smaller values 
    ai = 0.75/sqrt(4.0*p^2-1.0);
    aj = 0.75/sqrt(4.0*p^2-1.0);
    
%     ai = sqrt((2.0*p-1.0)/(2.0*p+1.0));
%     aj = sqrt((2.0*p-1.0)/(2.0*p+1.0));
    
    cmod = c;
    
    switch(p)
        case 0
%             disp('Moment Limiter not necessary for degree 0...');
            break;
        case 1
            % we first limit the highest order coefficient in the minmod
            % sense by comparing against differences of the next lowest
            % order coefficients. c(4) corresponds to phi(x,y) = x*y
            V = [c(4), aj*(cU(3)-c(3)), aj*(c(3)-cD(3)), ...
                       ai*(cR(2)-c(2)), ai*(c(2)-cL(2))];
            cmod(4) = minmod(V);
                           
            % if the highest order coefficient was modified, proceed to
            % also modify the two linear basis functions 
            if (abs(cmod(4)-c(4)) > tol)
                V = [c(3), aj*(cR(1)-c(1)), aj*(c(1)-cL(1))];
                cmod(3) = minmod(V);
                
                V = [c(2), ai*(cU(1)-c(1)), ai*(c(1)-cD(1))];
                cmod(2) = minmod(V);
            end
            
        case 2
            % we first limit the highest order coefficient in the minmod
            % sense by comparing against differences of the next lowest
            % order coefficients. c(4) corresponds to the x^2*y^2 basis
            % function
            V = [c(9), aj*(cU(8)-c(8)), aj*(c(8)-cD(8)), ...
                       ai*(cR(6)-c(6)), ai*(c(6)-cL(6))];
            cmod(9) = minmod(V);
            
            if (abs(cmod(9)-c(9)) > tol)
                % limit the coefficients corresponding to the x^2*y and
                % x*y^2 basis functions
                V = [c(8), aj*(cU(7)-c(7)), aj*(c(7)-cD(7)), ...
                           ai*(cR(5)-c(5)), ai*(c(5)-cL(5))];
                cmod(8) = minmod(V);
                
                V = [c(6), aj*(cU(5)-c(5)), aj*(c(5)-cD(5)), ...
                           ai*(cR(5)-c(3)), ai*(c(3)-cL(3))];
                cmod(6) = minmod(V);
                
                if (abs(cmod(8)-c(8)) > tol || abs(cmod(6)-c(6)) > tol)
                    % limit the coefficients corresponding to the x^2 and
                    % y^2 basis functions
                    V = [c(3), aj*(cU(2)-c(2)), aj*(c(2)-cD(2))];
                    cmod(3) = minmod(V);
                    
                    V = [c(7), aj*(cR(4)-c(4)), aj*(c(4)-cL(4))];
                    cmod(7) = minmod(V);
                    
                    if (abs(cmod(3)-c(3)) > tol || abs(cmod(7)-c(7)) > tol)
                        % limit the coefficients corresponding to the x*y
                        % basis function
                        V = [c(5), aj*(cU(4)-c(4)), aj*(c(4)-cD(4)), ...
                                   ai*(cR(2)-c(2)), ai*(c(2)-cL(2))];
                        cmod(5) = minmod(V);
                        
                        if (abs(cmod(5)-c(5)) > tol)
                            % limit the coefficients corresponding to the x
                            % and y basis functions
                            V = [c(2), aj*(cU(1)-c(1)), aj*(c(1)-cD(1))];
                            cmod(2) = minmod(V);

                            V = [c(4), aj*(cR(1)-c(1)), aj*(c(1)-cL(1))];
                            cmod(4) = minmod(V);
                        end
                    end
                end
            end
    end
    
    u(k,:) = cmod';
    
end

return


function m = minmod(V)
    
    if (~any(diff(sign(V(V~=0)))))
        m = sign(V(1))*min(abs(V));
    else
        m = 0.0;
    end

return

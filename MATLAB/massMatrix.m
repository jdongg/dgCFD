function M = massMatrix(p)

Nloc = (p+1)^2;
M = zeros(Nloc,Nloc);

[w,x,y] = quadRule2D(p+1);
phi = basisFunctions(p,x,y);

for i=1:Nloc
    for j=1:Nloc
        M(i,j) = sum(w'.*phi(i,:).*phi(j,:));
    end
end

return
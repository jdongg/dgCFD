function phiEdge = basisFunctionsEdge(p)

[w,x] = quadRule1D(p+1);

% evaluate basis functions on left edge (relative to current cell)
xk = -1.0*ones(length(w),1);
yk = x;
phiEdge.Left = basisFunctions(p,xk',yk');

% evaluate basis functions on bottom edge (relative to current cell)
xk = x;
yk = -1.0*ones(length(w),1);
phiEdge.Down = basisFunctions(p,xk',yk');

% evaluate basis functions on right edge (relative to current cell)
xk = ones(length(w),1);
yk = x;
phiEdge.Right = basisFunctions(p,xk',yk');

% evaluate basis functions on top edge (relative to current cell)
xk = x;
yk = ones(length(w),1);
phiEdge.Up = basisFunctions(p,xk',yk');

return
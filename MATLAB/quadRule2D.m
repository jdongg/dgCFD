function [w,x,y] = quadRule2D(Norder)

switch(Norder)
    case 1        
        wTmp = 2.0;
        xTmp = 0.0;
    case 2
        wTmp = [1.0; 1.0];
        xTmp = [-0.577350269189;
                 0.577350269189];
    case 3
        wTmp = [0.555555555555;
                0.888888888888;
                0.555555555555];
        xTmp = [-0.774596669241;
                 0.0;
                 0.774596669241];
        
    case 4
        wTmp = [0.347854845137;
                0.652145154862;
                0.652145154862;
                0.347854845137];
        xTmp = [-0.861136311954;
                -0.339981043584;
                 0.339981043584;
                 0.861136311954];
    case 5
        wTmp = [0.568888888888888;
                0.236926885056189;
                0.478628670499366;
                0.478628670499366;
                0.236926885056189];
        xTmp = [0.0;
               -0.906179845938664;
               -0.538469310105683;
                0.538469310105683;
                0.906179845938664];
end

[X,Y] = meshgrid(xTmp,xTmp);
[WX,WY] = meshgrid(wTmp,wTmp);

x = X(:);
y = Y(:);
w = WX.*WY;
w = w(:);
        
return
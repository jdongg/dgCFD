function [w,x] = quadRule1D(Norder)

switch(Norder)
    case 1        
        w = 2.0;
        x = 0.0;
    case 2
        w = [1.0; 1.0];
        x = [-0.577350269189;
              0.577350269189];
    case 3
        w = [0.555555555555;
             0.888888888888;
             0.555555555555];
        x = [-0.774596669241;
              0.0;
              0.774596669241];
        
    case 4
        w = [0.347854845137;
             0.652145154862;
             0.652145154862;
             0.347854845137];
        x = [-0.861136311954;
             -0.339981043584;
              0.339981043584;
              0.861136311954];
    case 5
        w = [0.568888888888888;
             0.236926885056189;
             0.478628670499366;
             0.478628670499366;
             0.236926885056189];
        x = [0.0;
            -0.906179845938664;
            -0.538469310105683;
             0.538469310105683;
             0.906179845938664];
end

return
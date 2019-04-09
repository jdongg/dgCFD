clear all; close all; clc;

x0 = linspace(-3,6,40); % create 200 equally spaced points between =2 and 6
x1 = linspace(-5,5,40); % create 200 equally spaced points between -3 and 3

myode = @(t,y) [y(2); -y(1)-y(1)^2/6 + 1/3*cos(3/5*t)];
    
t0 = 0;
tfinal = 20.0;

iter = 1;
for i=1:length(x0)
    for j=1:length(x1)
        [t,y] = ode45(myode, [t0 tfinal], [x0(i) x1(j)]);
        
        % test whether solution is unbounded
        if (abs(y(end,1)) > 40)
            % mark as unbounded
            marker(i,j) = 1;
            plot(x0(i),x1(j),'bx');
        else
            % mark as unbounded
            marker(i,j) = 0;
        end
        
        
        hold on;
        
        iter = iter+1;
    end
end

% [X0,X1] = 
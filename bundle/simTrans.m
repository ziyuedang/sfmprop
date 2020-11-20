function [a,b,dx,dy] = simTrans(xObs, yObs, xc, yc)
p = length(xObs);
% Design matrix
A = zeros(2*p,4);
for i = 1:p
    A(2*i-1,:) = [xc(i) -yc(i) 1 0];
    A(2*i,:) = [yc(i) xc(i) 0 1];
end

% Observation vector
l = zeros(2*p,1);
for i = 1:p
    l(2*i-1) = xObs(i);
    l(2*i) = yObs(i);
end

% Solve for unknowns (x) - Linear parameters
x = lscov(A,l);
a = x(1);
b = x(2);
dx = x(3);
dy = x(4);
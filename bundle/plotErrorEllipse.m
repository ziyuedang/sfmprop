function plotErrorEllipse(x,y,cov)
[eigenvec, eigenval] = eig(cov);
% ev1 -> large eigenvalue, ev2 -> small eigenvalue
ev1 = eigenval(1,1);
ev2 = eigenval(2,2);
if ev1 < 0 && ev2 < 0
    ev1 = -ev1;
    ev2 = -ev2;
end
if ev1 < ev2
    tmp = ev1;
    ev1 = ev2;
    ev2 = tmp;
end
if ev1 <= 0 || ev2 <= 0
    ev1 = abs(ev1);
    ev2 = abs(ev2);
    disp('Eigenvalue of Hessian is negative or zero (!)');
end
vx = eigenvec(1,1);
vy = eigenvec(2,1);
% Calculate axes
ax = sqrt(ev1);
by = sqrt(ev2);
% Calculate the angle between the x-axis and the largest eigenvector
angle = atan2(vx, vy);
% This angle is between -pi and pi.
% Let's shift it such that the angle is between 0 and 2pi
if(angle < 0)
    angle = angle + 2*pi;
end
% Get the 95% confidence interval error ellipse
%chisquare_val = 2.4477;
theta_grid = linspace(0,2*pi);
phi = angle;
%a   = chisquare_val*ax;
%b   = chisquare_val*by;
% the ellipse in x and y coordinates
ellipse_x_r  = ax*cos( theta_grid );
ellipse_y_r  = by*sin( theta_grid );
%Define a rotation matrix
R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];
%let's rotate the ellipse to some angle phi
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;
% Draw the error ellipse
plot(r_ellipse(:,1) + x,r_ellipse(:,2) + y,'y-','LineWidth',1.25)
hold on;
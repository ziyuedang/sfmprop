%% Single photo resection

% Transform lidar coordinates to sfm 3D point unitless coordinates

img =[2961, 5178, 8657, 6730; 1111, 1217, 2575, 366];
width = 8984;
height = 6732;
x = [-78.52, -75.86,-10.41, -121.14] + 273700;
y = [ -311.13, -203.58, -36.62, -145.35] + 3289700;
lidar_orig = [x;y; -7.71, -13.16, -7.21, -16.85];

s = 1;
R_sfm2lidar = [-3.695, 383.316, -5.996; 383.360, -3.668, -1.715; -1.658, -6.012, -383.33];
t = [273746.985, 3289700-241.142, 472.254]';

trythis = [0.5697, -0.4134, 1.2796]';
test = R_sfm2lidar * trythis + t;
lidar = zeros(size(lidar_orig));
coord = [];
for i = 1:size(lidar_orig,2)
    coord = lidar_orig(:,i);
    lidar(:,i) = inv(R_sfm2lidar) * (coord - t);
end
xp = 4476.76;
yp = 3363.07;
c = 11797.603;
l = [img(1, :), img(2, :)]';


%% Determine initial parameters
omega0 = 0;
phi0 = 0;
[a, b, deltaX, deltaY] = simTrans(lidar(1,:), lidar(2, :), img(1,:), img(2, :));
kappa0 = atan2(b,a);
xc0 = deltaX;
yc0 = deltaY;
zc0 = c*sqrt(a^2+b^2)+mean(lidar(3,:));


%%
initial = [xc0, yc0, zc0, omega0, phi0, kappa0]';
syms o p k xc yc zc X Y Z
% Form collinearity equation
M = rotxyz(o,p,k);
Uij = M(1,1)*(X-xc) + M(1,2)*(Y-yc) + M(1,3)*(Z-zc);
Wij = M(3,1)*(X-xc) + M(3,2)*(Y-yc) + M(3,3)*(Z-zc);
Vij = M(2,1)*(X-xc) + M(2,2)*(Y-yc) + M(2,3)*(Z-zc);
x = xp - c*Uij/Wij;
y = yp - c*Vij/Wij;
% Use jacobian function to get partial derivatives wrt each unknown
xJacob = jacobian(x,[xc, yc, zc, o, p, k]);
yJacob = jacobian(y,[xc, yc, zc, o, p, k]);
% Initialize and preallocate matrices
A = zeros(2*size(lidar, 2),6);
dx = ones(6,1);
est = zeros(2*size(lidar, 2),1); 
iter = 0;
maxIter = 10; %maximum # of iterations

coordTol = 0.05; tiltTol = 0.00056; kTol = 0.00053;


%% Try bagdad
photo = img';

xp = photo(:,1);
yp = photo(:,2);
f = c;
ng = size(photo,1);
XYZ = lidar';
omega = 0;
phi = 0;
kappa = 0;
xo = xc0;
yo = yc0;
zo = zc0;

disp('INITIAL EXTERIOR ORIENTATION')
wpk=[omega,phi,kappa,xo,yo,zo];
[omega,phi,kappa,xo,yo,zo] 
  
[ Tx, Ty, Tz, w2, p2, k2 ]= Imageresection (XYZ,xp,yp,wpk,f );
%% Start iteration
while max(abs(dx(1:3)))>=coordTol || abs(dx(5))>=tiltTol || abs(dx(6)) >= kTol
    if iter >= maxIter - 1
        break
    end
    %
    for i = 1:size(lidar,2)
        XYZ = lidar(:,1);
        [dxExt, dyExt, dxXYZ, dyXYZ] = jacobian(XYZ, initial, c);
        A(2*i - 1,:) = dxExt;
        A(2*i,:) = dyExt;
    end

    W = est-l;
    P = eye(2*size(lidar,2));
%    P(1:2:end-1,:) = P(1:2:end-1,:).*(1/sigObsX^2);
%    P(2:2:end,:) = P(2:2:end,:).*(1/sigObsY^2);
    dx = -inv(A'*A)*A'*W;
    V = A*dx + W;
    initial = initial + dx;
    iter = iter + 1;
end
resection = initial;
resectionDisp = [initial(1:3);rad2deg(initial(4:6))]';
disp('Exterior orientation parameters from resection: ')
disp(resectionDisp)

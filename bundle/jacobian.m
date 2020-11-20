function [dxExt, dyExt, dxXYZ, dyXYZ] = jacobian(XYZ_3D, initial, c)
    % Jacobian of x, y wrt exterior orientations (Xc, Yc, Zc, omega, phi, kappa)
    
    Xc = initial(1);
    Yc = initial(2);
    Zc = initial(3);
    o = initial(4);
    p= initial(5);
    k = initial(6);
    R= rotxyz(o, p, k);
    r11 = R(1,1);
    r12 = R(1,2);
    r13 = R(1,3);
    r21 = R(2,1);
    r22 = R(2,2);
    r23 = R(2,3);
    r31 = R(3,1);
    r32 = R(3,2);
    r33 = R(3,3);

    X = XYZ_3D(1);
    Y = XYZ_3D(2);
    Z = XYZ_3D(3);
    U = r11*(X - Xc) + r12*(Y - Yc) + r13*(Z - Zc);
    V = r21*(X - Xc) + r22*(Y - Yc) + r23*(Z - Zc);
    W = r31*(X - Xc) + r32*(Y - Yc) + r33*(Z - Zc);
    dUdPhi = -sin(p) * cos(k) * (X - Xc) + sin(o) * cos(p) * cos(k) * (Y - Yc) + cos(p) * cos(o) * cos(k) * (Z - Zc);
    dWdPhi = -cos(p) * (X - Xc) - sin(o) * sin(p) * (Y - Yc) - cos(o) * sin(p) * (Z - Zc);
    dVdPhi = -sin(p) * sin(k) * (X - Xc) + sin(o) * cos(p) * sin(k) * (Y - Yc) + cos(p) * sin(k) * sin(o) * (Z - Zc);
    
    
    dxXc = (c) * (r31 * U - r11 * W) / (W.^2);
    dxYc = (c) * (r32 * U - r12 * W) /(W.^2);
    dxZc = (c) * (r33 * U - r13 * W) / (W.^2);
    dxOmega = (c/(W.^2))* ((W * r13 - U * r33)* (Y - Yc) - (W * r12 - U * r32)* (Z - Zc));
    dxPhi = (c/(W.^2)) * (W * dUdPhi - U* dWdPhi);
    dxKappa = -c * (r21 * (X - Xc) + r22 * (Y - Yc) + r23 * (Z - Zc))/W;
    
    dxX = -dxXc;
    dxY = -dxYc;
    dxZ = -dxZc;
    
    dyXc = c * (V * r31 - W * r21)/W.^2;
    dyYc = c * (V * r32 - W * r22)/W.^2;
    dyZc = c *  (V * r33 - W * r23)/W.^2;
    dyOmega = (c /W.^2) * ((W * r23 - V * r33)* (Y - Yc) - (W * r22 - V * r32) * (Z - Zc));
    dyPhi =  (c /W.^2) * (W * dVdPhi - V * dWdPhi);
    dyKappa =  c * (r11 * (X - Xc) + r12 * (Y - Yc) + r13 * (Z - Zc))/W;
   
    dyX = -dyXc;
    dyY = -dyYc;
    dyZ = -dyZc;
    
    dxExt = [dxXc, dxYc, dxZc, dxOmega, dxPhi, dxKappa];
    dyExt = [dyXc, dyYc, dyZc, dyOmega, dyPhi, dyKappa];
    
    dxXYZ = [dxX, dxY, dxZ];
    dyXYZ = [dyX, dyY, dyZ];
    
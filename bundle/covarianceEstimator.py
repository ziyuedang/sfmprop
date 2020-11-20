# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:22:36 2020

Covariance estimate for results given by openMVG bundle adjustment.
One extra iteration is performed on the results, thus giving us the ability to 
compute covariance for parameters and 
@author: zdang2
"""
import sfm_IO as IO
import numpy as np
import math as m
from numpy.linalg import multi_dot
import scipy
import scipy.io as SIO
import scipy.sparse.linalg as linalg
from numpy.linalg import inv
import transforms3d

path = 'E:/SFM/wfh/Data/campus_p_0.12_ANNL2/reconstruction_global/'
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = IO.read_sfm('sfm2.json', path)
# root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = IO.read_sfm('sfm_georeferenced.json', 'E:/SFM/wfh/Data/campus_p_0.12_ANNL2/reconstruction_global/')
jacobian_filename = 'jacobian_num_rows_21492num_cols_9426.txt'
path_jacobian = 'E:/SFM/wfh/Data/campus_p_0.12_ANNL2/'
# covariance of observations - covxx, covxy, covyy
cov_l = IO.read_cov('cov_filenames.txt', 'E:/SFM/wfh/Data/campus/')

# load scaled covariance matrix for observations
C_l = np.load('C_l_scaled_campus.npy')
def invRotate(M):
    """
    Return omega, phi, kappa in radians given rotation matrix
    """
    o = m.atan2(M[2][1], M[2][2])   # pitch
    p = m.atan2(-M[2][0], m.sqrt(M[2][1]**2 + M[2][2]**2))  # yaw
    k = m.atan2(M[1][0], M[0][0])  # roll
    return o, p, k

def distance(lst, value):
    """
    Returns the index of the point in the array that matches the value
    """
    dist = []
    for i in range(len(lst)):
        dist.append((value[0] - lst[i, 0])**2 + (value[1] - lst[i, 1])**2)
        
    return np.argmin(dist)

def initParams(views_meta, coords_3d, extrinsics, intrinsics):
    """
    This function initializes bundle adjustment parameters and data inputs.
    Returns: Design matrices Ae, Ao, EO initial values (XYZc, rotation)
    2D image observations (merged_obs), 3D object point coordinates (XYZ_3D)
    """
    img_obs = []
    img_obs.clear()
    n_imgs = len(views_meta)

    # XYZ_3D - flattened 3D object point coordinates
    XYZ_3D = np.empty((0, 0))
    for i in range(0, len(coords_3d)):
        coords_3d[i]['index_3D'] = i
        XYZ_3D = np.append(XYZ_3D, np.array(coords_3d[i]['X']))

    for i in range(0, len(coords_3d)):
        pnt_3d = coords_3d[i]['observations']
        index_3d = coords_3d[i]['index_3D']  
        for img in pnt_3d:
            img.update({"index_3D": index_3d})
        img_obs.extend(pnt_3d)


    # list of 2-D image observations for all images
    # merged_obs is a list with length of (n_imgs), each value for each image
    # corresponds to number of observations for that image, (x, y, index of the 3D
    # point which can be found in coords_3d)
    merged_obs = []   
    obs_count = []
    feat_id = []
    merged_obs.clear()
    obs_count.clear()
    for i in range(0, n_imgs):
        temp = []
        temp.clear()
        feat_id_temp = []
        for d in range(0, len(img_obs)):
            keyID = img_obs[d]['key']
            if keyID == i:
                a = img_obs[d]['value']['x']
                b = img_obs[d]['index_3D']
                a.append(b)
                temp.append(a)
                feat_id_temp.append(img_obs[d]['value']['id_feat'])
        obs_count.append(len(temp))
        merged_obs.append(temp)
        feat_id.append(feat_id_temp)


    # Number of 2D image observations
    n = 2 * sum(obs_count) 
    
    # Number of exterior parameters
    ue = 6 * n_imgs
    
    # Number of 3D object points
    uo = 3 * len(coords_3d)
    
    # dxe, dxo
    dxe = np.ones((ue, 1))
    
    dxo = np.ones((uo, 1))
    
    # Exterior - Xc, Yc, Zc, Rotation
    exterior = np.empty([n_imgs, 6])
    for i in range(0, n_imgs):
        Xc, Yc, Zc = extrinsics[i]['center']
        o, p, k = invRotate(np.array(extrinsics[i]['rotation']))
        exterior[i, :] = np.array([Xc, Yc, Zc, o, p, k])
    exterior = exterior.flatten()
    # Intrinsics unpack
    interior = intrinsics['ptr_wrapper']['data']
     
    return dxe, dxo, exterior, merged_obs, XYZ_3D, interior, feat_id, n, ue, uo, obs_count

def compute_r_coeff(projected_point, k1, k2, k3):
    """
    Return: r_coefficient
    """
    r2 = np.inner(projected_point, projected_point)
    r4 = r2 * r2
    r6 = r4 * r2
    r = 1 + k1 * r2 + k2 * r4 + k3 * r6
    return r

def apply_distortion(interior, transformed_3D_point):
    k1, k2, k3 = interior['disto_k3']
    c = interior['focal_length']
    xp, yp = interior['principal_point']
    
    # 3D homogeneous -> 2D euclidean
    projected_point = np.array(transformed_3D_point[0:2]) / transformed_3D_point[-1]
    
    # distortion
    r_c = compute_r_coeff(projected_point, k1, k2, k3)
    
    return xp, yp, projected_point, r_c, c
    
        
def projected_2D(exterior, XYZ_3D, interior, merged_obs):
    """
    This evaluates the image x, y based on collinearity.
    Input: interior, exterior, 3D object point coordinates
    Return: x, y on the image (pixel)
    """ 
    
    # [x, y] = K * R(XYZ - XYZc)
    estimate = []
    for i in range(0, len(merged_obs)):
        XYZc = exterior[i*6:(i*6+3)]
        o, p, k = exterior[(i*6+3):(i+1)*6]
        for j in range(0, len(merged_obs[i])):
            xyz_3D_idx = merged_obs[i][j][2]
            XYZ = XYZ_3D[xyz_3D_idx*3:(xyz_3D_idx+1)*3]
            R = transforms3d.euler.euler2mat(o, p, k)
            transformed_3D_point = np.dot(R, XYZ) - np.dot(R, XYZc)  
            xp, yp, projected_point, r_coeff, c = apply_distortion(interior, transformed_3D_point)
            # collinearity equations
            # x = xp - c*(m11*(X - Xc) + m12*(Y - Yc) + m13*(Z - Zc))/(m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc))
            # y = yp - c*(m21*(X - Xc) + m22*(Y - Yc) + m23*(Z - Zc))/(m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc))
            estimate_x = xp + projected_point[0] * r_coeff * c
            estimate_y = yp + projected_point[1] * r_coeff * c
            estimate.append(estimate_x)
            estimate.append(estimate_y)
    estimate_array = np.array(estimate)

    return estimate_array
    
def Jacobian(XYZ_3D, exterior, interior):
    """
    Jacobian of x, y wrt exterior orientations (Xc, Yc, Zc, omega, phi, kappa)
    """
    Xc, Yc, Zc, o, p, k = exterior
    _, _, _, r_coeff, c = apply_distortion(interior, XYZ_3D)
    R = transforms3d.euler.euler2mat(o, p, k)
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    X, Y, Z = XYZ_3D
    
    U = r11*(X - Xc) + r12*(Y - Yc) + r13*(Z - Zc)
    V = r21*(X - Xc) + r22*(Y - Yc) + r23*(Z - Zc)
    W = r31*(X - Xc) + r32*(Y - Yc) + r33*(Z - Zc)
    dUdPhi = -m.sin(p) * m.cos(k) * (X - Xc) + m.sin(o) * m.cos(p) * m.cos(k) * (Y - Yc) + m.cos(p) * m.cos(o) * m.cos(k) * (Z - Zc)
    dWdPhi = -m.cos(p) * (X - Xc) - m.sin(o) * m.sin(p) * (Y - Yc) - m.cos(o) * m.sin(p) * (Z - Zc)
    dVdPhi = -m.sin(p) * m.sin(k) * (X - Xc) + m.sin(o) * m.cos(p) * m.sin(k) * (Y - Yc) + m.cos(p) * m.sin(k) * m.sin(o) * (Z - Zc)
    
    
    dxXc = (c * r_coeff) * (r31 * U - r11 * W) / W**2
    dxYc = (c * r_coeff) * (r32 * U - r12 * W) / W**2
    dxZc = (c * r_coeff) * (r33 * U - r13 * W) / W**2
    dxOmega = (c * r_coeff/W**2) * ((W * r13 - U * r33) * (Y - Yc) - (W * r12 - U * r32) * (Z - Zc))
    dxPhi = (c * r_coeff/W**2) * (W * dUdPhi - U * dWdPhi)
    dxKappa = -c * r_coeff * (r21 * (X - Xc) + r22 * (Y - Yc) + r23 * (Z - Zc))/W
    
    dxX = -dxXc
    dxY = -dxYc
    dxZ = -dxZc
    
    dyXc = c * r_coeff * (V * r31 - W * r21)/W**2
    dyYc = c * r_coeff * (V * r32 - W * r22)/W**2
    dyZc = c * r_coeff * (V * r33 - W * r23)/W**2
    dyOmega = (c * r_coeff/W**2) * ((W * r23 - V * r33) * (Y - Yc) - (W * r22 - V * r32) * (Z - Zc))
    dyPhi =  (c * r_coeff/W**2) * (W * dVdPhi - V * dWdPhi)
    dyKappa =  c * r_coeff * (r11 * (X - Xc) + r12 * (Y - Yc) + r13 * (Z - Zc))/W
    
    dyX = -dyXc
    dyY = -dyYc
    dyZ = -dyZc
    
    dxExt = np.array([dxXc, dxYc, dxZc, dxOmega, dxPhi, dxKappa])
    dyExt = np.array([dyXc, dyYc, dyZc, dyOmega, dyPhi, dyKappa])
    
    dxXYZ = np.array([dxX, dxY, dxZ])
    dyXYZ = np.array([dyX, dyY, dyZ])
    
    return dxExt, dyExt, dxXYZ, dyXYZ

def get_Gp(XYZ_3D, uo):
    """
    This function forms matrix Gp. G = (Gp, Gs).T
    """
    X = XYZ_3D[0::3]
    Y = XYZ_3D[1::3]
    Z = XYZ_3D[2::3]
    GpT = np.empty((7, 0))
    for i in range(int(uo/3)):
        temp = np.array([[1,     0,    0],
                         [0,     1,    0],
                         [0,     0,    1],
                         [0,    -Z[i], Y[i]],
                         [Z[i],  0,   -X[i]],
                         [-Y[i], X[i], 0],
                         [X[i],  Y[i], Z[i]]])
        GpT = np.append(GpT, temp, axis = 1)
    Gp = GpT.T
    return Gp
    
def get_Gs(exterior, ue):
    """
    This function forms matrix Gs. G = (Gp, Gs).T
    """
    X = exterior[0::6]
    Y = exterior[1::6]
    Z = exterior[2::6]
    GsT = np.empty((7, 0))
    for i in range(int(ue/6)):
        temp = np.array([[1,     0,    0,    0, 0, 0],
                         [0,     1,    0,    0, 0, 0],
                         [0,     0,    1,    0, 0, 0],
                         [0,    -Z[i], Y[i], 1, 0, 0],
                         [Z[i],  0,   -X[i], 0, 1, 0],
                         [-Y[i], X[i], 0,    0, 0, 1],
                         [X[i],  Y[i], Z[i], 0, 0, 0]])
        GsT = np.append(GsT, temp, axis = 1)
    Gs = GsT.T
    return Gs

def get_cov(merged_obs, cov_l, n, scale):
    """
    Return the covariance matrix of observations.
    """
    C_l = np.zeros((n, n))
    n_obs_count = 0
    for img in range(0, len(merged_obs)):
        up_bound = n_obs_count
        n_pnts = len(merged_obs[img])
        print(n_pnts)
        n_obs_count += n_pnts * 2
        C_img = np.zeros((n_pnts * 2, n_pnts * 2))
        lst = cov_l[img]
        for i in range(0, n_pnts):
            # check if index returned from searching x/y is the same
            value = merged_obs[img][i][0], merged_obs[img][i][1]
            idx = distance(lst, value)
            cov_xx = scale ** 2 * cov_l[img][idx, 2]
            # cov_xy = scale ** 2 * cov_l[img][idx, 3]
            cov_yy = scale ** 2 * cov_l[img][idx, 4]
            C_img[(i*2):(2*i+2), (i*2):(2*i+2)] = ([cov_xx, 0], [0, cov_yy])
        C_l[up_bound:n_obs_count, up_bound:n_obs_count] = C_img
    return C_l

def get_tol(ue, uo, coordsTol, kTol, tiltTol):
    """
    This function generates tolerance t same length as unknowns
    """
    n_imgs = int(ue/6)
    tExt = np.empty((ue, 1))
    t3d = np.ones((uo, 1)) * coordsTol
    for i in range(n_imgs):
        tExt[6*i: 6*i+3] = coordsTol
        tExt[6*i+3: 6*i+5] = tiltTol
        tExt[6*i+5] = kTol   
    
    t = np.concatenate((tExt, t3d))
    return t

def lsq_constrained(A, P, G, w, bool_sparse):
    """
    This function calculates (dx, lambda)' = -inv(ATPA, G; G.T, 0)*(ATPw, 0).T
    where lambda is the Lagrange multipliers (not of interest)
    Returns dx
    """
    ATPA = multi_dot([A.T, P, A])
    ATPw = multi_dot([A.T, P, w])
    ATPw = np.reshape(ATPw, (ATPw.shape[0], 1))
    N = np.concatenate((np.concatenate((ATPA, G), axis = 1), np.concatenate((G.T, np.zeros((7, 7))), axis = 1)), axis = 0)
    n =  (-1) * np.concatenate((ATPw, np.zeros((7, 1))))
    if bool_sparse == True:
        sparseN = scipy.sparse.csc_matrix(N)
        Cx_sparse = scipy.sparse.linalg.inv(sparseN)
        Cx = scipy.sparse.csc_matrix.toarray(Cx_sparse)
    else:
        Cx = inv(N)
    dx_lambda =  np.dot(Cx, n)
    dx = dx_lambda[0:len(dx_lambda) - 7]
    dx = np.ravel(dx)
    return dx, Cx

def get_l(merged_obs):
    """
    Return the observation vector
    """
    l = np.array([])
    for img in merged_obs:
        for pnt in img:
            l_x = pnt[0]
            l_y = pnt[1]
            l = np.append(l, [l_x, l_y], axis=0)
            
    return l

def get_A(interior, XYZ_3D, exterior, merged_obs, ue, uo):
    img_cnt = 0
    n_obs_cnt = 0
    A_e = np.empty([n, ue])
    A_o = np.empty((0, uo))
    for img in merged_obs:
        up_bound = n_obs_cnt
        n_obs_cnt += len(img) * 2
        exterior_img = exterior[img_cnt * 6 : (img_cnt + 1)*6]
        img_cnt += 1
        dExt_img = np.empty((0, 6))
        dXYZ_img = np.empty([len(img) * 2, uo])
        n_obs_img = 0
        for obs in img:
            
            xyz_3D_pnt_ID = obs[2]
            xyz_3D_pnt = coords_3d[xyz_3D_pnt_ID]['X']
            
            # Retrieve jacobian for 1 2-D point
            dxExt_pnt, dyExt_pnt, dxXYZ_pnt, dyXYZ_pnt = Jacobian( xyz_3D_pnt, exterior_img, interior)
            
            # Combine x and y
            dExt_pnt = np.array([dxExt_pnt, dyExt_pnt]) # 2 rows of dx/dExterior and dy/dExterior
            dXYZ_pnt = np.array([dxXYZ_pnt, dyXYZ_pnt]) # 2 rows of dx/dXYZ and dy/dXYZ
            
            # Construct part of the Ae/Ao for 1 image 
            dExt_img = np.append(dExt_img, dExt_pnt, axis=0)
            dXYZ_img[n_obs_img*2:(n_obs_img+1)*2, (3*xyz_3D_pnt_ID):(3*(xyz_3D_pnt_ID+1))] = dXYZ_pnt
            
            n_obs_img += 1
        
        A_e[up_bound:n_obs_cnt, ((img_cnt-1)*6):(img_cnt*6)] = dExt_img
        A_o = np.append(A_o, dXYZ_img, axis=0)
    A = np.concatenate((A_e, A_o), axis = 1)
    return A

def leastSquare(initial, interior, XYZ_3D, merged_obs, n_imgs, ue, uo, l, C_l, dx, sparse):
    """
    Bundle Adjustment
    """
    iteration = 0
    maxIter = 30
    while (np.absolute(dx) >= tol).any():
        if iteration >= maxIter:
            break
        
        # Split the initial to exterior and OPC
        exterior = initial[0:n_imgs*6]
        XYZ = initial[n_imgs*6::]
        
        # Get A
        # A = get_A(interior, XYZ_3D, exterior, merged_obs, ue, uo)

        # The order of parameters are different:
        # rotation angle[0], rotation angle[1], rotation angle[2], Xc, Yc, Zc
        A = IO.read_jacobian(jacobian_filename, path_jacobian, n_imgs)
        # Get Gp
        Gp = get_Gp(XYZ, uo)
        
        # Get Gs
        Gs = get_Gs(exterior, ue)
        
        # Combine Gp and Gs
        G = np.concatenate((Gp, Gs))

        # Get estimate
        estimate = projected_2D(exterior, XYZ, interior, merged_obs)
        
        # Get w
        # w = estimate - l
        w = np.squeeze(IO.read_residuals('residuals21492.txt', path_jacobian))
        
        if sparse == True:
            # weights of observations
            sparseC_l = scipy.sparse.csc_matrix(C_l)
            P_sparse = scipy.sparse.linalg.inv(sparseC_l)
            P = scipy.sparse.csc_matrix.toarray(P_sparse)
        else:
            P = inv(C_l)
        
        # dx = inv(A'PA)*A'Pw
        dx, Cx= lsq_constrained(A, P, G, w, sparse)
        initial += dx
        iteration += 1

    # Remove G
    Cx = Cx[0:-7, 0:-7]

    # Calculate stds
    diag = Cx.diagonal()
    diag_new = diag[0:(ue + uo)]
    std = np.sqrt(diag_new)
    
    # Convert radians to degrees
    # for i in range(0, int(ue/6)):
    #    initial[6 * i + 3 : 6 * (i + 1)] = np.degrees(initial[6 * i + 3 : 6 * (i + 1)])
    #    std[6 * i + 3 : 6 * (i + 1)] = np.degrees(std[6 * i + 3 : 6 * (i + 1)])
        
    # Calculate residuals
    V = np.dot(A, dx) + w

    # Separate Vx and Vy
    # Vx = V[::2]
    # Vy = V[1::2]

    
    # Reference variance
    m = len(l) # number of functions
    n = uo + ue # number of unknowns
    refVar = multi_dot([V.T, P, V])/(m - n)

    # Adjusted C_l
    # C_l_adjusted = multi_dot([A, Cx, A.T])

    
    return initial, V, P, std, refVar, iteration, l, Cx, obs_count, C_l_adjusted, m, n

def scaleCov(C_l, refVar):
    return C_l * refVar

# Initialize parameter blocks
scale = 0.000006
scale = 1
dxe, dxo, exterior, merged_obs, XYZ_3D, interior, feat_id, n, ue, uo, obs_count = initParams(views_meta, coords_3d, extrinsics, intrinsics)
n_imgs = int(ue/6)
# Observation vector
l = get_l(merged_obs)
    
# Observation covariance (before correction)


C_l = get_cov(merged_obs, cov_l, n, scale)

# Use scaled C_l instead
# C_l = np.load('scaled_C_l.npy')
# C_l = np.load('scaled_C_l_V_based.npy')
# Get tolerance
tol = get_tol(ue, uo, 0.0001, 0.0005, 0.0005)
    
 # Enter least square iteration
initial = np.concatenate((exterior, XYZ_3D))
dx = np.concatenate((dxe, dxo))

# Enter first round of least square iterations
sparse = True

# C_l = C_l_scaled
initial, V, P, std, refVar, iteration, l, Cx, obs_count, C_l_adjusted, m, n = leastSquare(initial, interior, XYZ_3D, merged_obs, n_imgs, ue, uo, l, C_l, dx, sparse)
C_l_scaled = scaleCov(C_l, abs(refVar))
C_x_scaled = scaleCov(Cx, abs(refVar))


# Georeferencing error propagation
# X_new = S * R * X + t
# cov_X_new = S^2 * R * cov_X * R'
Cx_3D = C_x_scaled[ue::, ue::]
cov_X_new = np.empty((Cx_3D.shape))

# Georefencing results
s_georef = 383.171
R_georef = np.array([[-0.0107815, 0.999818, -0.0157572], [0.999941, 0.0107594, -0.00148813], [-0.00131832, -0.0157723, -0.999875]])
t = np.array([[273747.01], [3289457.931], [471.297]])

# Georegistration with als 
s_georef = 1
R_georef = np.array([[-3.695, 383.316, -5.996], [383.360, 3.668, -1.715], [-1.658, -6.012, -383.330]])
t = np.array([[46.985], [-241.142], [472.254]])

for i in range(len(coords_3d)):
    Cx_temp = Cx_3D[i*3:(i+1)*3, i*3:(i+1)*3]
    cov_X_new[i * 3 : (i + 1) * 3, i *3 :(i+1)*3] = (s_georef ** 2) * multi_dot([R_georef, Cx_temp, R_georef.T])


XYZ_ref = np.empty((uo, 1))


for i in range(len(coords_3d)):
    XYZ_temp = np.array([[XYZ_3D[3*i]], [XYZ_3D[3*i+1]], [XYZ_3D[3*i+2]]])
    XYZ_ref[3*i:3*(i+1)] = s_georef * np.dot(R_georef, XYZ_temp) + t
# Save to mat file
SIO.savemat('campus_scaled_cx.mat', {'Cx': C_x_scaled})
SIO.savemat('C_l_orig_campus.mat',{'C_l_orig':C_l})
SIO.savemat('campus_original.mat', {'initial':initial, 'V':V, 'refVar':refVar,
'obs':l, 'obs_count':obs_count})
SIO.savemat('campus_scaled.mat',{'initial':initial, 'V':V, 'refVar':refVar, 'XYZ_georef':XYZ_ref, 
'obs':l, 'cov_unk_3D':cov_X_new, 'obs_count':obs_count})

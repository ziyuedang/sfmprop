# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:26:28 2019
This is the version 2 of bundle adjustment. 
@author: Ziyue Dang
"""

import numpy as np
import math as m
from scipy.linalg import inv
from numpy.linalg import multi_dot

def read_data(file_name):
    """
    Reads data from txt, returns camera parameters, initial estimates of point coordinates
    in the world frame, initial estimates of exterior orientations, 
    camera indices involved in each observation, point indices, measured 2D coordinates,
    number of images, number of points on each image
    """
    file = open(file_name, "r")
    # Apply int function to the file as a iterable, return iterators
    n_cameras, n_points, n_observations, n_imgs = map(int, file.readline().split())
    
    # Initialize arrays
    camera_indices = np.empty(n_observations, dtype = int)
    point_indices = np.empty(n_observations, dtype = int)
    points_2d = np.empty((n_observations, 2))
    n_pnts_each = np.empty(n_imgs, dtype = int)
    obs_weights = np.empty((n_observations, 4))
    
    for i in range(n_imgs):
        n_pnts_each[i] = int(file.readline())
    
    for i in range(n_observations):
        camera_index, point_index, x, y, Pxx, Pxy, Pyx, Pyy = file.readline().split()
        camera_indices[i] = int(camera_index)
        point_indices[i] = int(point_index)
        points_2d[i] = [float(x), float(y)]
        obs_weights[i] = [float(Pxx), float(Pxy), float(Pyx), float(Pyy)]
    points_2d = points_2d * 0.001
    point_indices = point_indices.reshape((n_observations, 1))    
    camera_params = np.empty(n_cameras * 3)   # 3 camera parameters [c, xp, yp]
    for i in range (n_cameras * 3):
        camera_params[i] = float(file.readline())
    camera_params = camera_params.reshape((n_cameras, -1))
    
    initialExterior = np.empty(n_imgs * 6)
    for i in range(n_imgs * 6):
        initialExterior[i] = float(file.readline())
    initialExterior = initialExterior.reshape((n_imgs, -1))
    # convert angles from degrees to radians
    initialExterior[:, 3:6] = np.radians(initialExterior[:, 3:6])
    
    points_3d = np.empty(n_points * 3)
    for i in range (n_points * 3):
        points_3d[i] = float(file.readline())
    points_3d = points_3d.reshape((n_points, -1))
    
    flag_control = np.empty(n_points * 3)
    for i in range (n_points * 3):
        flag_control[i] = int(file.readline())
    flag_control = flag_control.reshape((n_points, -1))
   
    return camera_params, points_3d, initialExterior, point_indices, points_2d, n_imgs, n_pnts_each, flag_control, obs_weights

def initializeParams(camera_params, points_3d, initialExterior, point_indices, points_2d, n_imgs, n_pnts_each, flag_control):
    """
    This function initializes (formats) the parameters that are used in least square.
    Returns: Ae, Ao, dxe, dxo, l, initialExterior, points_3d, point_indices_each.
    """
    n = 2*points_2d.shape[0]
    uo = np.count_nonzero((1 - flag_control))
    ue = 6*n_imgs
    Ae = np.empty([n, ue])
    Ao = np.empty([n, uo])
    dxe = np.ones((ue, 1))
    dxo = np.ones((uo, 1))
    initialExterior = initialExterior.flatten().T
    flag_control = flag_control.flatten().T
    points_3d = points_3d.flatten().T
    initial3d = points_3d[np.nonzero(1 - flag_control)] 
    split_index = np.array([1]*(n_imgs-1))
    tempSum = 0
    for k in range(n_pnts_each.shape[0] - 1):
        temp = n_pnts_each[k]
        tempSum += temp
        split_index[k] = tempSum
    point_indices_each = np.split(point_indices, split_index)
    
    
    return Ae, Ao, dxe, dxo, initialExterior, initial3d, points_3d, point_indices_each, uo, ue

def rotate(omega, phi, kappa):
    """
    Takes 3 rotational angles in radians, and return the rotation matrix M  
    """
    a = m.sin(omega)
    b = m.cos(omega)
    c = m.sin(phi)
    d = m.cos(phi)
    e = m.sin(kappa)
    f = m.cos(kappa)
    M = np.array([[d*f, b*e + a*c*f, a*e - b*c*f],
                  [-d*e, b*f - a*c*e, a*f + b*c*e],
                  [c, -a*d, b*d]])
    
    return M

def collinearity(camera_params, exterior, points_3d):
    """
    This evaluates the image x, y based on collinearity.
    Input: camera parameters, exterior orientation parameters, points' coordinates
    in real world frame.
    Return: x, y on the image (pixel)
    """
    Xc, Yc, Zc, omega, phi, kappa = exterior
    # X, Y, Z object point coordinates
    X, Y, Z = points_3d
    # camera parameters
    c = camera_params[:, 0]
    xp = camera_params[:, 1]
    yp = camera_params[:, 2]
    # rotation matrix
    M = rotate(omega, phi, kappa)
    m11 = M[0][0]
    m12 = M[0][1]
    m13 = M[0][2]
    m21 = M[1][0]
    m22 = M[1][1]
    m23 = M[1][2]
    m31 = M[2][0]
    m32 = M[2][1]
    m33 = M[2][2]
    # collinearity equations
    x = xp - c*(m11*(X - Xc) + m12*(Y - Yc) + m13*(Z - Zc))/(m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc))
    y = yp - c*(m21*(X - Xc) + m22*(Y - Yc) + m23*(Z - Zc))/(m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc))
    
    return x, y

def Jacobian(camera_params, exterior, points_3d):
    """
    This function calculates the partial derivatives of x, y wrt unknowns
    Return: dx2Ext(x wrt exterior unknowns), dx2XYZ(x wrt 3D object points)
            dy2Ext(y wrt exterior unknowns), dy2XYZ(x wrt 3D object points)
    """
    Xc, Yc, Zc, omega, phi, kappa = exterior
    c = camera_params[:, 0]
    X, Y, Z = points_3d
    M = rotate(omega, phi, kappa)
    m11 = M[0][0]
    m12 = M[0][1]
    m13 = M[0][2]
    m21 = M[1][0]
    m22 = M[1][1]
    m23 = M[1][2]
    m31 = M[2][0]
    m32 = M[2][1]
    m33 = M[2][2]   
    U = m11*(X - Xc) + m12*(Y - Yc) + m13*(Z - Zc)
    W = m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc)
    V = m21*(X - Xc) + m22*(Y - Yc) + m23*(Z - Zc)
    # derivatives of x wrt exterior and 3D object points
    dxXc = -(c/W**2)*(m31*U - m11*W)
    dxYc = -(c/W**2)*(m32*U - m12*W)
    dxZc = -(c/W**2)*(m33*U - m13*W)
    dxOmega = -(c/W**2)*((Y - Yc)*(m33*U - m13*W) - (Z - Zc)*(m32*U - m12*W))
    dxPhi = -(c/W**2)*((X - Xc)*(-W*m.sin(phi)*m.cos(kappa) - U*m.cos(phi)) 
    + (Y - Yc)*(W*m.sin(omega)*m.cos(phi)*m.cos(kappa) - U*m.sin(omega)*m.sin(phi))
    + (Z - Zc)*(-W*m.cos(omega)*m.cos(phi)*m.cos(kappa) + U*m.cos(omega)*m.sin(phi)))
    dxKappa = -c*V/W
    dxX = -dxXc
    dxY = -dxYc
    dxZ = -dxZc
    # derivatives of y wrt exterior and 3D object points
    dyXc = -(c/W**2)*(m31*V - m21*W)
    dyYc = -(c/W**2)*(m32*V - m22*W)
    dyZc = -(c/W**2)*(m33*V - m23*W)
    dyOmega = -(c/W**2)*((Y - Yc)*(m33*V - m23*W) - (Z - Zc)*(m32*V - m22*W))
    dyPhi = -(c/W**2)*((X - Xc)*(W*m.sin(phi)*m.sin(kappa) - V*m.cos(phi)) 
    + (Y - Yc)*(-W*m.sin(omega)*m.cos(phi)*m.sin(kappa) - V*m.sin(omega)*m.sin(phi))
    + (Z - Zc)*(W*m.cos(omega)*m.cos(phi)*m.sin(kappa) + V*m.cos(omega)*m.sin(phi)))
    dyKappa = c*U/W
    dyX = -dyXc
    dyY = -dyYc
    dyZ = -dyZc
    
    dx2Ext = np.array([dxXc, dxYc, dxZc, dxOmega, dxPhi, dxKappa])
    dx2XYZ = np.array([dxX, dxY, dxZ])
    dy2Ext = np.array([dyXc, dyYc, dyZc, dyOmega, dyPhi, dyKappa])
    dy2XYZ = np.array([dyX, dyY, dyZ])
    
    return dx2Ext, dx2XYZ, dy2Ext, dy2XYZ


def get_estimate(camera_params, exterior, points_3d, point_indices_each, n_pnts_each, n_imgs):
    """
    This retrieves the estimate of x, y image coordinates based on the collinearity equation
    using the updated unknown parameters.
    """
    
    estimate = np.array([])
    for j in range(n_imgs):
        for i in range(point_indices_each[j].shape[0]):
            k = np.squeeze(point_indices_each[j][i]) # k is the actual index of the point iterated to
            x, y = collinearity(camera_params, exterior[6*j:(6*j + 6)], points_3d[3*(k - 1):3*k])
            tempXY = np.array([x, y])
            estimate = np.append(estimate, tempXY)
    return estimate

def get_Ae(camera_params, exterior, points_3d, points_2d, point_indices_each, n_pnts_each, n_imgs): 
    """
    This forms the Ae matrix (partial derivatives taken wrt exterior parameters).
    """
    Ae = np.empty((points_2d.shape[0] * 2, 0))    
    n_pnts_iterated = 0
    for j in range(n_imgs):
        # each AeEachImg corresponds to the design matrix component for that image
        AeEachImg = np.zeros((points_2d.shape[0] * 2, 6)) 
        for i in range(point_indices_each[j].shape[0]):           
            k = np.squeeze(point_indices_each[j][i]) # k is the actual index of the point iterated to
            extTemp = exterior[6 * j : 6 * (j + 1)]
            points_3dTemp = points_3d[3 * (k - 1) : 3 * k]
            dx2Ext, _, dy2Ext, _ = Jacobian(camera_params, extTemp, points_3dTemp)
            tempDxDy = np.transpose(np.hstack((dx2Ext, dy2Ext)))
            AeEachImg[n_pnts_iterated * 2 : (n_pnts_iterated + 1) * 2] = tempDxDy
            n_pnts_iterated += 1
        Ae = np.append(Ae, AeEachImg, axis = 1)   
    
    return Ae

def get_Ao(camera_params, exterior, points_3d, points_2d, point_indices_each, n_pnts_each, n_imgs, flag_control):
    """
    This forms the Ao matrix (partial derivatives taken wrt object point unknowns)
    """
    n_control = int(np.count_nonzero((flag_control.flatten()))/3)
    print("n_control = ", n_control)
    uo = np.count_nonzero((1 - flag_control.flatten()))
    Ao = np.empty((0, uo))
    for j in range(n_imgs):
        n_pnts_iterated = -1 # this refers to image points
        AoEachImg = np.zeros((n_pnts_each[j] * 2, uo))
        for i in range(point_indices_each[j].shape[0]):
            k = np.squeeze(point_indices_each[j][i])
            # raise the number of points iterated even when it's a control point
            n_pnts_iterated += 1 
            if flag_control[(k - 1), 0] == 1:
                continue
            extTemp = exterior[6 * j : 6 * (j + 1)]
            points_3dTemp = points_3d[3 * (k - 1) : 3 * k]
            _, dx2XYZ, _, dy2XYZ = Jacobian(camera_params, extTemp, points_3dTemp)
            tempDxDy = np.transpose(np.hstack((dx2XYZ, dy2XYZ)))
            tempIdx = k - n_control
            AoEachImg[n_pnts_iterated * 2 : (n_pnts_iterated + 1) * 2, \
                      (tempIdx - 1) * 3 : tempIdx * 3] = tempDxDy
          
        Ao = np.append(Ao, AoEachImg, axis = 0)
    return Ao
def get_Gp(points_3d, uo):
    """
    This function forms matrix Gp. G = (Gp, Gs).T
    """
    X = points_3d[0::3]
    Y = points_3d[1::3]
    Z = points_3d[2::3]
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
    
def get_l(points_2d):
    """
    This function returns observation vector l.
    """
    l = points_2d.flatten().T
    return l

def get_w(estimate, l):
    """
    This function returns w with input of estimate and l
    """
    w = estimate - l
    return w

def get_P(obs_weights):
    """
    This function generates weight matrix P with input n (number of observations)
    """
    n_2d_pnts = obs_weights.shape[0]
    P = np.zeros((2 * n_2d_pnts, 2 * n_2d_pnts))
    obs_weights = np.reshape(obs_weights, (2 * n_2d_pnts, 2))
    for i in range (n_2d_pnts):
        P[2 * i : 2 * (i + 1), 2 * i : 2 * (i + 1)] = obs_weights[2 * i : 2 * (i + 1)]
    return P

def get_tol(n_imgs, n_pnts_total, coordsTol, kTol, tiltTol, flag_control):
    """
    This function generates tolerance t same length as unknowns
    """
    n_control = int(np.count_nonzero((flag_control.flatten())))
    tExt = np.empty([n_imgs * 6])
    for j in range(n_imgs):
        tExt[6*j: 6*j+3] = coordsTol
        tExt[6*j+3: 6*j+5] = tiltTol
        tExt[6*j+5] = kTol
    
    t3d = np.full((n_pnts_total - n_control), coordsTol)
    
    t = np.concatenate((tExt, t3d))
    return t

def lsq(A, P, w):
    """
    This function calculates dx = -inv(A'PA)A'Pw
    """
    ATPA = multi_dot([A.T, P, A])    
    invATPA = inv(ATPA)
    dx = - multi_dot([invATPA, A.T, P, w])
    return dx

def lsq_constrained(A, P, G, w):
    """
    This function calculates (dx, lambda)' = -inv(ATPA, G; G.T, 0)*(ATPw, 0).T
    where lambda is the Lagrange multipliers (not of interest)
    Returns dx
    """
    ATPA = multi_dot([A.T, P, A])
    ATPw = multi_dot([A.T, P, w])
    ATPw = np.reshape(ATPw, (ATPw.shape[0], 1))
    N = np.concatenate((np.concatenate((ATPA, G), axis = 1), np.concatenate((G.T, np.zeros((7, 7))), axis = 1)), axis = 0)
    n =  - np.concatenate((ATPw, np.zeros((7, 1))))
    dx_lambda =  np.dot(inv(N), n)
    dx = dx_lambda[0:len(dx_lambda) - 7]
    dx = np.ravel(dx)
    return N, dx

def solve_bundle(file_name, maxIter):
    """
    This function is the main function to solve for bundle adjustment..
    Returns solved parameters and their covariances.
    """
    # Read data
    camera_params, points_3d, initialExterior, point_indices, points_2d, \
    n_imgs, n_pnts_each, flag_control, obs_weights = read_data(file_name)
    
    # Initialize Parameters
    Ae, Ao, dxe, dxo, initialExterior, initial3d, points_3d, point_indices_each, uo, ue \
    = initializeParams(camera_params, points_3d, initialExterior, \
                       point_indices, points_2d, n_imgs, n_pnts_each, flag_control)
    
    # Get observation vector l
    l = get_l(points_2d)
    
    # Get P
    P = get_P(obs_weights)
    
    # Get tolerance
    tol = get_tol(n_imgs, points_3d.shape[0], 0.01, 0.000005, 0.000005, flag_control)
    
    # Enter least square iteration
    initial = np.concatenate((initialExterior, initial3d))
    dx = np.concatenate((dxe, dxo))
    iteration = 0
    
    while (np.absolute(dx) >= tol).any():
        if iteration >= maxIter:
            break
        # Split the initial to exterior and OPC
        exterior = initial[0:n_imgs*6]
        unkOPC = initial[n_imgs*6::]
        # Get unknown point indices
        unk_ind = np.nonzero((1 - flag_control.flatten()))
        
        # Update unknown OPCs to all 3d points
        points_3d[unk_ind] = unkOPC
        
        # Get Ae
        Ae = get_Ae(camera_params, exterior, points_3d, points_2d, point_indices_each, n_pnts_each, n_imgs)
        
        # Get Ao
        Ao = get_Ao(camera_params, exterior, points_3d, points_2d, point_indices_each, n_pnts_each, n_imgs, flag_control)
        
        # Combine Ae and Ao
        A = np.concatenate((Ae, Ao), axis = 1)
        
        # Get Gp
        Gp = get_Gp(points_3d, uo)
        
        # Get Gs
        Gs = get_Gs(exterior, ue)
        
        # Combine Gp and Gs
        G = np.concatenate((Gp, Gs))

        # Get estimate
        estimate = get_estimate(camera_params, exterior, points_3d, point_indices_each, n_pnts_each, n_imgs)
        
        # Get w
        w = get_w(estimate, l)

        # dx = inv(A'PA)*A'Pw
        N, dx = lsq_constrained(A, P, G, w)
        initial += dx
        iteration += 1

    # Calculate stds
    Cx = inv(N)
    diag = Cx.diagonal()
    diag_new = diag[0:(ue + uo)]
    std = np.sqrt(diag_new)
    
    # Convert radians to degrees
    for i in range(int(ue/6)):
        initial[6 * i + 3 : 6 * (i + 1)] = np.degrees(initial[6 * i + 3 : 6 * (i + 1)])
        std[6 * i + 3 : 6 * (i + 1)] = np.degrees(std[6 * i + 3 : 6 * (i + 1)])
        
    # Calculate residuals
    V = np.dot(A, dx) + w
    
    # Reference variance
    m = len(l) # number of functions
    n = uo + ue # number of unknowns
    refVar = multi_dot([V.T, P, V])/(m - n)
    
    Cl = multi_dot([A, Cx, A.T])
    
    
    return initial, V, P, std, refVar, iteration, l, Cx, Cl
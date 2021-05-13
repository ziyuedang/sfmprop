# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:54:18 2020

Data I/O.

@author: zdang2
"""
import json
import pandas as pd
import numpy as np
import scipy.io as SIO

def read_sfm(file_name, path):
    """
    Reads SfM results generated by openMVG and parse the data.
    Input: filename, path.
    Output: metadata - views, internal parameters - intrinsics,
            external parameters - extrinsics, 3-D point coordinates - sturcture
            control points
    """
    file = path + file_name
    with open(file, 'r') as openfile:
        sfm = json.load(openfile)

    control_points = sfm['control_points']
    extrinsics = sfm['extrinsics']
    intrinsics = sfm['intrinsics']
    root_path = sfm['root_path']
    coords_3d = sfm['structure']
    views_meta = sfm['views']

    # Formatting
    intrinsics = intrinsics[0]['value']
    
    for i in range(0, len(extrinsics)):
        extrinsics[i] = extrinsics[i]['value']
    
    for i in range(0, len(views_meta)):
        views_meta[i] = views_meta[i]['value']
        
    for i in range(0, len(coords_3d)):
        coords_3d[i] = coords_3d[i]['value']
#    
    return root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points

def save_sfm(path, file_name, XYZ_BA, rotation_BA, center_BA):
    file = path + file_name
    size_xyz = int(len(XYZ_BA)/3)
    with open(file, 'r') as openfile:
        sfm = json.load(openfile)
    
    out_file_name = path + 'sfm_BA.json'
    out_file = open(out_file_name, 'w')
    for i in range(size_xyz):
        temp = XYZ_BA[i*3:(i+1) * 3]
        a = temp.tolist()
        sfm['structure'][i]['value']['X'] = a
    
    for i in range(int(len(center_BA)/3)):
        temp = rotation_BA[i].tolist()
        sfm['extrinsics'][i]['value']['rotation'] = temp
        sfm['extrinsics'][i]['value']['center'] = center_BA[i * 3 : (i + 1) * 3]
    json.dump(sfm, out_file, indent=4)
    out_file.close()
    return None 
    

def read_cov(cov_filenames, path):
    """
    Reads covariance data corresponding to each image and their key points.
    Format: covxx, covxy, covyy
    """
    file_names_file = path + cov_filenames
    file_names_pds = pd.read_csv(file_names_file, sep = ';', header=None, engine='python')
    file_names = file_names_pds.values.tolist()
    cov = []
    for file_name in file_names[0]:
        file = path + file_name
        content = []
        df = pd.read_csv(file, header=None, skiprows = [0, 1], sep = '\t')
        content = df.to_numpy()        
        cov.append(content[:, 0:5])
            
            
    return cov      

def read_jacobian(jacobian_filename, path_jacobian, n_imgs):
    """
    Read jacobian data provided by ceres solver.
    """
    ue = 6 * n_imgs
    filename = path_jacobian + jacobian_filename
    with open(filename, 'r') as openfile:
        content = pd.read_csv(openfile, sep = '\s+', header=None)
    
    A = content.to_numpy()
    return A
    
def read_residuals(residual_filename, path):
    """
    Read residuals? (w) Misclosure to be exact. (estimated value - meausured value)
    """
    filename = path + residual_filename
    with open(filename, 'r') as openfile:
        residuals = pd.read_csv(openfile, header=None)
    w = residuals.to_numpy()
    return w
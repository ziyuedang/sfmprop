"""
Lidar 3D data back projection - takes lidar data and camera parameters, generate corresponding
images for each image plane.
"""
import numpy as np
from numpy import linalg
from laspy.file import File
from sfm_IO import read_sfm
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image
import math as m
import cv2
import statistics as stat
# Utils

def load_lidar(filename):
    inFile = File(filename, mode = 'r')
    point = inFile.points
    return inFile, point

# Projections

def pos_3d_2_image_res(intrinsics, extrinsics_single_img, width, height, pos_3d_point):
    # Apply external parameters
    cam_R = extrinsics_single_img['rotation']
    center = extrinsics_single_img['center']
    t = - np.dot(cam_R, np.transpose(np.asarray([center]))) 
    transformed_point = np.dot(cam_R, pos_3d_point) + t
    # Transform the point from homogeneous to euclidean
    eucli_x  = transformed_point[0]/transformed_point[2]
    eucli_y  = transformed_point[1]/transformed_point[2]
    projected_point = np.array([[eucli_x], [eucli_y]])

    # Apply intrinsic parameters
    focal = intrinsics['ptr_wrapper']['data']["focal_length"]
    px = intrinsics['ptr_wrapper']['data']["principal_point"][0]
    py = intrinsics['ptr_wrapper']['data']["principal_point"][1]
    k1 = intrinsics['ptr_wrapper']['data']["disto_k3"][0]
    k2 = intrinsics['ptr_wrapper']['data']["disto_k3"][1]
    k3 = intrinsics['ptr_wrapper']['data']["disto_k3"][2]

    # Apply distortion
    r2 = m.sqrt(projected_point[0]**2 + projected_point[1]**2)
    r4 = r2 * r2
    r6 = r4 * r2
    r_coeff = 1 + k1 * r2 + k2 * r4 + k3 * r6
    projected_x = px + (projected_point[0] * r_coeff) * focal
    projected_y = py + (projected_point[1] * r_coeff) * focal
    return projected_x.flatten(), projected_y.flatten()

def lidar_2_pos_3d(lidar, intrinsics, extrinsics_single_img):
    # correct offset
    x_offset = 273700
    y_offset = 3289700
    lidar_corr = np.array([lidar[0] - x_offset, lidar[1] - y_offset, lidar[2]])

    R_georef = np.array([[-3.695, 383.316, -5.996], [383.360, 3.668, -1.715], [-1.658, -6.012, -383.330]])
    t_georef = np.array([[46.985], [-241.142], [472.254]])

    X_lidar2sfm = np.dot(np.linalg.inv(R_georef), (lidar_corr - t_georef))

    return X_lidar2sfm

# import lidar data
inFile, point_records = load_lidar("E:/SFM/wfh/UH_campus/ALS/Titan_data.las")
x = np.asarray(inFile.x)
y = np.asarray(inFile.y)
z = np.asarray(inFile.z)
intensity = np.asarray(inFile.intensity)

# import camera meta data
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = read_sfm("sfm_data.json", "E:/SFM/wfh/Data/campus_p_0.12_ANNL2_new/reconstruction_global/")
width = intrinsics['ptr_wrapper']['data']["width"]
height = intrinsics['ptr_wrapper']['data']["height"]

pixel_sz = 0.03 # size of pixel in meters

n_imgs = len(coords_3d)
pixel_coords_all_imgs = []
x_min = 0
y_min = 0
x_max = width
y_max = height
imgID = 0
extrinsics_single_img = extrinsics[imgID]
X_m = []
Y_m = []
index = []
sfm_z = []
for j in tqdm(range(0, len(x))):
    lidar = np.transpose(np.array([[x[j], y[j], z[j]]]))
    X_lidar2sfm = lidar_2_pos_3d(lidar, intrinsics, extrinsics_single_img)
    X_pixel, Y_pixel = pos_3d_2_image_res(intrinsics, extrinsics_single_img, width, height, X_lidar2sfm)
    if X_pixel > x_min and X_pixel < x_max and Y_pixel > y_min and Y_pixel < y_max:
        X_m.append(X_pixel)
        Y_m.append(Y_pixel)
        sfm_z.append(X_lidar2sfm[2])
        index.append(j)
    else:
        continue

intensity_new = intensity[index]
norm_intensity = 255*(intensity_new/5000)
# image processing
# def fillPixel(img, x, y, intensity):
#    img[x][y] = intensity/5000
    
img = cv2.imread("E:/SFM/wfh/Data/campus/11.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pixelCoords = np.frompyfunc(list, 0, 1)(np.empty((x_max, y_max), dtype=object))
# pixelCoords = np.full((x_max, y_max, 0), [])
for i in range(0, len(X_m)):
    x_pixel = m.floor(X_m[i])
    y_pixel = m.floor(Y_m[i])
    # print(x_pixel)
    # print(y_pixel)
    if len(pixelCoords[x_pixel][y_pixel]) == 0:
       pixelCoords[x_pixel][y_pixel] = [i]
    else:
       pixelCoords[x_pixel][y_pixel].append(i)

new_img = np.zeros((height, width), np.uint8)
for i in range(0, x_max):
    for j in range(0, y_max):
        pointID = pixelCoords[i][j]
        if len(pointID) == 0 :
            continue
        elif len(pointID) == 1:
            new_img[j][i] = norm_intensity[pointID]
        else:
            tempZ = []
            for pID in pointID:
                tempZ.append(sfm_z[pID][0])
            filteredID = [z for z in tempZ[z] if (tempZ[z] - min(tempZ)) < 0.0026]
            filteredIntensity = norm_intensity[filteredID]
            new_img[j][i] = stat.mean(filteredIntensity)





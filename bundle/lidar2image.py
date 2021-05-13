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
from scipy.interpolate import griddata
from skimage import transform
import scipy.spatial as spatial
from os import listdir
import os

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
    return projected_x.item(), projected_y.item()

def lidar_2_pos_3d(lidar, intrinsics, extrinsics_single_img):
    # correct offset
    x_offset = 273700
    y_offset = 3289700
    lidar_corr = np.array([lidar[0] - x_offset, lidar[1] - y_offset, lidar[2]])

    R_georef = np.array([[-3.695, 383.316, -5.996], [383.360, 3.668, -1.715], [-1.658, -6.012, -383.330]])
    t_georef = np.array([[46.985], [-241.142], [472.254]])

    X_lidar2sfm = np.dot(np.linalg.inv(R_georef), (lidar_corr - t_georef))

    return X_lidar2sfm

# import camera meta data
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = read_sfm("sfm_data.json", "E:/SFM/wfh/Data/campus_p_0.12_ANNL2_new/reconstruction_global/")
width = intrinsics['ptr_wrapper']['data']["width"]
height = intrinsics['ptr_wrapper']['data']["height"]

x_max = width
y_max = height
pixel_sz = 0.03 # size of pixel in meters
n_imgs = len(extrinsics)
orig_img_path = "E:/SFM/wfh/Data/campus/"

# import lidar data
lidar_path = "E:/SFM/wfh/UH_campus/ALS/"

# Generate intensity image for each perspective plane (camera angle)
inFile_lidar, point_records = load_lidar(lidar_path + "Titan_data.las")

# output path for intensity images

path_output = "E:/SFM/wfh/Data/titan_images/"

for imgID in range(0, n_imgs):
    img_filename = views_meta[imgID]['ptr_wrapper']['data']['filename']
    img = Image.open("E:/SFM/wfh/Data/campus/"+img_filename)
    newsize = (1497, 1122)
    im1 = img.resize(newsize)
    im1 = im1.save(path_output  + "Original_downsampled_small" + img_filename)
    # X_m, Y_m, sfm_z, index, intensity = pointsToImg(imgID, 4, inFile_lidar, width, height)
    # image, idx_lidar = formatImage(X_m, Y_m, sfm_z, index, x_max, y_max, intensity)
    # orig_img = cv2.cvtColor(cv2.imread("E:/SFM/wfh/Data/campus/" + img_filename), cv2.COLOR_BGR2GRAY)
    
    # orig_downsample, orig_downsampleSmall = downSample(y_max, x_max, orig_img)
    # idx_lidar_ds = downsampleIdx(x_max, y_max, idx_lidar)
    
    # cv2.imwrite(path_output + "Original_downsampled_small" + img_filename, orig_downsampleSmall)
    # downsampledImage, downsampleSmall = downSample(y_max, x_max, image)
#    cv2.imwrite(path_output + "Intensity_downsampled_small" + img_filename, downsampleSmall)

#    interpImage = image_iterp(downsampleSmall, downsampleSmall.shape[1], downsampleSmall.shape[0])
#    cv2.imwrite(path_output + "Intensity_IDW_filtered" + img_filename, interpImage)
#    np.save(path_output + '/output' + "/lidar_index_" + img_filename + ".npy", idx_lidar_ds)

# Read intensity images 

path_intensity_imgs = "E:/SFM/wfh/Data/titan_images/output/"
parent_dir = "E:/SFM/wfh/Data/titan_images/output"

for imgID in range(0, n_imgs):
    img_filename = views_meta[imgID]['ptr_wrapper']['data']['filename']
    intensity_img_filename = "Intensity_IDW_filtered" + img_filename
    idx_lidar_ds = np.load(path_output + '/output' + "/lidar_index_" + img_filename + '.npy', allow_pickle=True)
    intensity_img = cv2.cvtColor(cv2.imread(path_intensity_imgs + intensity_img_filename), cv2.COLOR_BGR2GRAY)
    #orig_img = cv2.cvtColor(cv2.imread(orig_img_path + img_filename), cv2.COLOR_BGR2GRAY)
    #orig_downsample, orig_downsampleSmall = downSample(y_max, x_max, orig_img)
    patchSize = 150
    iteration = 0
    path = os.path.join(parent_dir, img_filename)
    mode = 0o666
    #os.mkdir(path, mode)
    #print("Directory '%s' created" % img_filename)
    for i in range(0, intensity_img.shape[0], patchSize):
        for j in range(0, intensity_img.shape[1], patchSize):
            iteration += 1
            if (intensity_img.shape[0] - i) < patchSize and (intensity_img.shape[1] - j) < patchSize:
                #orig_patch = orig_downsampleSmall[i:intensity_img.shape[0], j:intensity_img.shape[1]]
                intensity_patch = intensity_img[i:intensity_img.shape[0], j:intensity_img.shape[1]]
                lidar_idx_patch = idx_lidar_ds[i:intensity_img.shape[0], j:intensity_img.shape[1]]
            elif (intensity_img.shape[0] - i) < patchSize and (intensity_img.shape[1] - j) >= patchSize:
                #orig_patch = orig_downsampleSmall[i:intensity_img.shape[0], j:(j+patchSize)]
                intensity_patch = intensity_img[i:intensity_img.shape[0], j:(j+patchSize)]
                lidar_idx_patch = idx_lidar_ds[i:intensity_img.shape[0], j:(j+patchSize)]
            elif (intensity_img.shape[0] - i) >= patchSize and (intensity_img.shape[1] - j) < patchSize:
                #orig_patch = orig_downsampleSmall[i:(i+patchSize), j:intensity_img.shape[1]]
                intensity_patch = intensity_img[i:(i+patchSize), j:intensity_img.shape[1]]
                lidar_idx_patch = idx_lidar_ds[i:(i+patchSize), j:intensity_img.shape[1]]
            else:
                #orig_patch = orig_downsampleSmall[i:(i+patchSize), j:(j+patchSize)]
                intensity_patch = intensity_img[i:(i+patchSize), j:(j+patchSize)]
                lidar_idx_patch = idx_lidar_ds[i:(i+patchSize), j:(j+patchSize)]
            currentPath = os.path.join(path, str(iteration))
            if os.path.isdir(currentPath):
                np.save(currentPath + '/lidar_index_' + str(iteration)+ "_" + img_filename + '.npy', lidar_idx_patch)
            else:
                os.mkdir(currentPath, mode)
                np.save(currentPath + '/lidar_index_' + str(iteration)+ "_" + img_filename + '.npy', lidar_idx_patch)
            #os.mkdir(currentPath, mode)
            #cv2.imwrite(currentPath + "/orignal_" + str(iteration)  + "_" + img_filename, orig_patch)
            #cv2.imwrite(currentPath + "/intensity_" + str(iteration) + "_" + img_filename, intensity_patch)
            

def dataParsing(inFile):
    x = np.asarray(inFile.x)
    y = np.asarray(inFile.y)
    z = np.asarray(inFile.z)
    intensity = np.asarray(inFile.intensity)
    return x, y, z, intensity

def pointsToImg(imgID, iterations, inFile, width, height):
    x, y, z, intensity = dataParsing(inFile)    
    x_min = 0
    y_min = 0
    x_max = width
    y_max = height
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
    return X_m, Y_m, sfm_z, index, intensity

def formatImage(X_m, Y_m, sfm_z, index, x_max, y_max, intensity):
    intensity_new = intensity[index]
    norm_intensity = intensity_new

    pixelCoords = np.frompyfunc(list, 0, 1)(np.empty((x_max,y_max), dtype=object))
    ind_pixelCoords = np.frompyfunc(list, 0, 1)(np.empty((y_max, x_max), dtype=object))
    for i in range(0, len(X_m)):
        x_pixel = m.floor(X_m[i])
        y_pixel = m.floor(Y_m[i])

        if len(pixelCoords[x_pixel][y_pixel]) == 0:
            pixelCoords[x_pixel][y_pixel] = [i]
        else:
            pixelCoords[x_pixel][y_pixel].append(i)

    new_img = np.zeros((height, width), np.uint16)
    num_single_point_pixel = 0
    num_pixels = 0
    emptyPixels = 0
    single_point = []
    multi_point = []
    for i in range(0, x_max):
        for j in range(0, y_max):
            pointID = pixelCoords[i][j]   
            if len(pointID) == 0 :
                emptyPixels += 1
                continue
            elif len(pointID) == 1:
                new_img[j][i] = norm_intensity[pointID]           
                num_single_point_pixel += 1
                single_point.append(new_img[j][i])

                ind_pixelCoords[j][i] = [index[pointID[0]]] # append index of lidar points
            else:
                intensityPixel = []
                temp = []
                for pID in pointID:
                    intensityPixel.append(norm_intensity[pID])
                    temp.append(index[pID])
                num_pixels += len(intensityPixel) 
                new_img[j][i] = m.floor(stat.mean(intensityPixel))
                multi_point.append(new_img[j][i])

                ind_pixelCoords[j][i] = temp

    allPixel = []
    for i in range(0, x_max):
        for j in range(0, y_max):
            pixelValue = new_img[j][i]
            allPixel.append(pixelValue)

    normalized_img = (new_img - new_img.min()) /(max(new_img.flatten()) - min(new_img.flatten())) * 255
    normalized_img_uint8 = normalized_img.astype(np.uint8)

    hist, bins = np.histogram(normalized_img_uint8.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img2 = cdf[normalized_img_uint8]
    return img2, ind_pixelCoords


# Downsampling image
def downSample(y_max, x_max, img2):
    n_pixels = 6
    img_downsampled = np.zeros((y_max, x_max))
    img_downsampled_small = np.zeros((int(y_max/6), int(x_max/6)))

    for i in range(0, x_max, n_pixels):
        for j in range(0, y_max, n_pixels):
            if (x_max - i) < n_pixels or (y_max - j) < n_pixels:
                continue
            else:
                sumPatch = 0
                cnt = 0
                for m in range(0, n_pixels):
                    for n in range(0, n_pixels):
                        if img2[j+n][i+m] == 0:
                            continue
                        else:
                            cnt += 1
                            sumPatch += img2[j+n][i+m]
                if cnt == 0:
                    avgPatch = 0
                else:
                    avgPatch = int(sumPatch/cnt)
                img_downsampled_small[int(j/6), int(i/6)] = avgPatch
                img_downsampled[j:(j+n_pixels), i:(i+n_pixels)] = avgPatch * np.ones((n_pixels, n_pixels))

    return img_downsampled, img_downsampled_small

def downsampleIdx(x_max, y_max, idx_lidar):
    n_pixels = 6
    idx_lidar_downsampled = np.frompyfunc(list, 0, 1)(np.empty((int(y_max/6), int(x_max/6)), dtype=object))
    for i in range(0, x_max, n_pixels):
        for j in range(0, y_max, n_pixels):
            if (x_max - i) < n_pixels or (y_max - j) < n_pixels:
                continue
            else:
                temp = []
                for k in range(0, n_pixels):
                    for p in range(0, n_pixels):
                        temp += idx_lidar[j + k, i + p]
                idx_lidar_downsampled[int(j/6), int(i/6)] = temp
    return idx_lidar_downsampled

'''
# Nearest neighbor to fill the remaining black pixels
def moving_average(img_downsampled, x_max, y_max, iterations):
    dummy_img = np.zeros((y_max, x_max))
    for i in range(1, x_max-1):
        for j in range(1, y_max-1):
            val = 0
            numNonBlack = 0
            if(img_downsampled[j][i] != 0):
                dummy_img[j][i] = img_downsampled[j][i]
                continue
            for x in range(0, 3):
                for y in range(0, 3): 
                    if (x-1) == 0 and (y-1) == 0:
                        continue
                    if img_downsampled[j+x-1][i+y-1] != 0:
                        numNonBlack += 1
                        val += img_downsampled[j+x-1][i+y-1]
            if(numNonBlack == 0):
                continue
            newVal = val / numNonBlack
            dummy_img[j][i] = newVal
    if iterations == 0:
        return dummy_img
    return moving_average(dummy_img, x_max, y_max, iterations -1)
'''
def distance(x1, y1, x2, y2):
    d = np.sqrt((x1 - x2)**2 + (y1- y2)**2)
    return d

# Inverse Distance Weighted algorithm
def idw_npoint(xz,yz, img_downsampled, x_max, y_max, r, p):
    z_block=[]
    xr_min=xz-r
    xr_max=xz+r
    yr_min=yz-r
    yr_max=yz+r
    if xr_min <= 0:
        xr_min = 0
    if xr_max >= x_max:
        xr_max = x_max
    if yr_min <= 0:
        yr_min = 0
    if yr_max >= y_max:
        yr_max = y_max
    nonZero_neighbors = 0
    w = []
    tempZ = 0
    for i in range(xr_min, xr_max):
        for j in range(yr_min, yr_max):
            d = distance(xz, yz, i, j)
            if d <= r and d > 0:                
                tempZ = img_downsampled[j][i]
                if tempZ != 0:
                    w.append(1/(d**p))
                    z_block.append(tempZ)
                    nonZero_neighbors += 1
    if nonZero_neighbors == 0:
        z_idw = 0    
    else:
        z_idw = np.dot(z_block, w)/sum(w)
    return z_idw

# Alternative solution - interpolation to fill the black pixels
def image_iterp(img_downsampled, x_max, y_max):
    idw_img = np.zeros((y_max, x_max))
    for i in tqdm(range(0, x_max)):
        for j in range(0, y_max):
            if img_downsampled[j][i] == 0:
                idw_pixelV = idw_npoint(i, j, img_downsampled, x_max, y_max, 3, 2)
                idw_img[j][i] = idw_pixelV                          
            else:
                idw_img[j][i] = img_downsampled[j][i]
    return idw_img

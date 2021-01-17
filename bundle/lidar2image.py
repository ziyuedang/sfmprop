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

def threedeetoImage(imgID, iterations):    
    img_filename = views_meta[imgID]['ptr_wrapper']['data']['filename']
    # pixel_coords_all_imgs = []
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
    image = formatImage(X_m, Y_m, sfm_z, index, x_max, y_max)
    cv2.imwrite("First" + img_filename, image) 
    downsampledImage = downSample(y_max, x_max, image)
    cv2.imwrite("FSecond" + img_filename, downsampledImage) 
    averageImage = moving_average(downsampledImage, x_max, y_max, iterations)
    cv2.imwrite("Final" + img_filename, averageImage) 

def formatImage(X_m, Y_m, sfm_z, index, x_max, y_max):
    intensity_new = intensity[index]
    # norm_intensity = intensity_new * 25/255
    norm_intensity = intensity_new
    # norm_intensity = 255*(intensity_new/5000)
    # image processing
    # def fillPixel(img, x, y, intensity):
    #    img[x][y] = intensity/5000
        
    # img = cv2.imread("E:/SFM/wfh/Data/campus/" + img_filename)

    pixelCoords = np.frompyfunc(list, 0, 1)(np.empty((x_max, y_max), dtype=object))
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
    single_point = []
    multi_point = []
    for i in range(0, x_max):
        for j in range(0, y_max):
            pointID = pixelCoords[i][j]
            if len(pointID) == 0 :
                continue
            elif len(pointID) == 1:
                new_img[j][i] = norm_intensity[pointID]
                num_single_point_pixel += 1
                single_point.append(new_img[j][i])
            else:
                intensityPixel = []
                for pID in pointID:
                    intensityPixel.append(norm_intensity[pID])
                num_pixels += len(intensityPixel) 
                new_img[j][i] = m.floor(stat.mean(intensityPixel))
                multi_point.append(new_img[j][i])

    allPixel = []
    for i in range(0, x_max):
        for j in range(0, y_max):
            pixelValue = new_img[j][i]
            allPixel.append(pixelValue)

    normalized_img = (new_img - new_img.min()) /(max(new_img.flatten()) - min(new_img.flatten())) * 255
    normalized_img_uint8 = normalized_img.astype(np.uint8)
    # equ = cv2.equalizeHist(normalized_img_uint8)
    # res = np.hstack((normalized_img_uint8, equ))
    hist, bins = np.histogram(normalized_img_uint8.flatten(),256,[0,256])

    cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img2 = cdf[normalized_img_uint8]
    return img2


 # Downsampling image
def downSample(y_max, x_max, img2):
    img_downsampled = np.zeros((y_max, x_max))
    n_pixels = 6
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
                img_downsampled[j:(j+n_pixels), i:(i+n_pixels)] = avgPatch * np.ones((n_pixels, n_pixels))
    return img_downsampled

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

# cv2.imwrite("DummythiCC" + img_filename, dummy_img)       
# cv2.imwrite("Downsampled" + img_filename, img_downsampled)
threedeetoImage(1, 4)
# for i in range(1, x_max-1):
#     for j in range(1, y_max-1):
#         prev = img_downsampled[j, i-1]
#         curr = img_downsampled[j, i]
#         next = img_downsampled[j, i+1]
#         if curr == 0:
#             if prev == 0:
#                 continue
#             elif prev != 0:
#                 curr = prev
#         else:
#             continue
#         img_downsampled[j, i] = curr
# cv2.imwrite("new.jpg", img_downsampled)

# for i in range(0, x_max):
# for j in range(0, y_max):
#     pointID = pixelCoords[i][j]
#     if len(pointID) == 0 :
#         continue
#     elif len(pointID) == 1:
#         new_img[j][i] = norm_intensity[pointID]
#     else:
#         tempZ = []
#         for pID in pointID:
#             tempZ.append(sfm_z[pID][0])
#         filteredID = [index for index, value in enumerate(tempZ) if (value - min(tempZ)) < 0.0026]
#         filteredIntensity = [norm_intensity[index] for index in filteredID]
#         new_img[j][i] = m.floor(stat.mean(filteredIntensity))

   # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(normalized_img_uint8.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
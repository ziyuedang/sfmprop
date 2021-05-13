"""
Script for analysis of covariance estimate - 
This evaluates the uncertainty for more key points instead of just the 3D Model points.
A large amount (300+) of total 3D points are needed for evaluation so the results can be binned into 
multiple distributions.
 
"""
import numpy as np
import os
import pandas as pd
import covarianceEstimator as covE
import sfm_IO as IO
import scipy
import scipy.io as SIO
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import curve_fit

path_matches = "E:/SFM/wfh/Data/titan_images/downsampled_photo/matches/geometric_export/"
path_results = "E:/SFM/wfh/Data/titan_images/results/"

# Read validation data
# validation_df = pd.read_csv(path_results + "titan_validation_points.csv", header=None) 
# sfm_3D_idx = validation_df[0].to_numpy()
# lidar_3d_x = validation_df[1].to_numpy()
# lidar_3d_y = validation_df[2].to_numpy()
# lidar_3d_z = validation_df[3].to_numpy()

# Read sfm data
path = 'E:/SFM/wfh/Data/titan_images/original/reconstruction_global/'
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = IO.read_sfm('sfm_registered.json', path)

# Read jacobian and residuals
n_imgs = len(views_meta)
jacobian_filename = 'jacobian_num_rows_24366num_cols_10578.txt'
path_jacobian = 'E:/SFM/wfh/Data/titan_images/original/reconstruction_global/'
jacobian = IO.read_jacobian(jacobian_filename, path_jacobian, n_imgs)
# remove the GCP observation columns in design matrix
jacobian = np.delete(jacobian, np.s_[-33::], axis = 1)
# remove the image observations for GCPs
# jacobian = np.delete(jacobian, np.s_[-106::], axis = 0)
residuals = IO.read_residuals('residuals24366.txt', path_jacobian)
# residuals = residuals[0:5712]

# Read covariance info for each image
cov_l = IO.read_cov('cov_filenames.txt', 'E:/SFM/wfh/Data/titan_images/original/cov_eval/')

#========================= Run covariance estimator to determine reference variance=================================

# Initialize parameter block
scale = 1
dxe, dxo, exterior, XYZ_3D, interior, ue, uo, l1, n_obs_per_img = covE.initParams(views_meta, coords_3d, extrinsics, intrinsics)

interior_params = np.array([interior['focal_length'], interior['principal_point'][0], \
    interior['principal_point'][1], interior['disto_k3'][0], \
    interior['disto_k3'][1], interior['disto_k3'][2]])

ui = 6
dui = np.zeros((6, 1))
l2 = []
GCP = []
for i in range(0, len(control_points)):
    control_obs = control_points[i]["value"]["observations"]
    GCP.extend(control_points[i]["value"]["X"])
    for j in control_obs:
        l2.extend(j["value"]["x"])

n_GCP_obs = len(l2)
C_l2 = np.zeros((len(l2), len(l2)))
np.fill_diagonal(C_l2, 2**2)

# Number of 2D image observations
n = len(l1)
# Observation vector & Observation covariance (before correction)
C_l1 = np.zeros((n, n))
temp = 0

for i in range(0, len(coords_3d)):
    struct_obs = coords_3d[i]["observations"]  
    for j in struct_obs:
        row_id = j["value"]["id_feat"] 
        covxx = cov_l[j["key"]][row_id, 2]
        covyy = cov_l[j["key"]][row_id, 4]
        C_l1[temp*2, temp*2] = abs(covxx)
        C_l1[(temp * 2 + 1), (temp * 2 + 1)] = abs(covyy)
        temp += 1

l = l1 + l2
# l = l1
placeholder = np.zeros((n, n_GCP_obs))

C_l = np.concatenate((np.concatenate((C_l1, placeholder), axis = 1), np.concatenate((placeholder.T, C_l2), axis = 1)), axis = 0)
# C_l = C_l1
# Tolerance
tol = covE.get_tol(ue, uo, 0.0001, 0.0005, 0.0005)

# Run least square iteration
use_interior = True
if (use_interior):
    initial = np.concatenate((exterior, interior_params, XYZ_3D))
    dx = np.concatenate((dxe, dui, dxo))
else:
    initial = np.concatenate((exterior, XYZ_3D))
    dx = np.concatenate((dxe, dxo)) 
    jacobian = np.delete(jacobian, np.s_[ue:(ue+ui)], axis = 1)




sparse = False
dx, V, refVar, scale, iteration, l, Cx, m, n = covE.leastSquare(initial, GCP, interior_params, XYZ_3D, n_imgs, ue, uo, ui, l, C_l1, C_l2, C_l, dx, sparse, residuals, jacobian, tol)

print("Reference Variance = ", refVar)

# scale C_l1 by reference variance
# C_l1_scaled = covE.scaleCov(C_l1_scaled, abs(scale))
C_l1_scaled = covE.scaleCov(C_l1, abs(1/refVar))
C_l2_scaled = covE.scaleCov(C_l2, abs(1/refVar))
C_l_scaled = covE.scaleCov(C_l, abs(1/refVar))



#=========================Run covariance estimator to compute the final covariance====================================
# Clear variables
# update C_l
del initial, dx

if (use_interior):
    initial = np.concatenate((exterior, interior_params, XYZ_3D))
    dx = np.concatenate((dxe, dui, dxo))
else:
    initial = np.concatenate((exterior, XYZ_3D))
    dx = np.concatenate((dxe, dxo)) 

dx, V, refVar, scale, iteration, l, Cx, m, n = covE.leastSquare(initial, GCP, interior_params, XYZ_3D,n_imgs, ue, uo, ui, l, C_l1_scaled, C_l2_scaled, C_l_scaled, dx, sparse, residuals, jacobian, tol)
print("After scaling, reference variance (should be 1) = ", refVar)

#==========================Georeferencing error propagation=========================================
if (use_interior):
    Cx_3D = Cx[(ue+ui)::, (ue+ui)::]
    C_ext = Cx[0:ue, 0:ue]
    XYZ_3D_new = XYZ_3D + dx[90::].flatten()
else:
    Cx_3D = Cx[ue::, ue::]
    C_ext = Cx[0:ue, 0:ue]
    XYZ_3D_new = XYZ_3D + dx[ue::].flatten()

# Extract estimated errors for exterior
angle1 = []
angle2 = []
angle3 = []
x_ext_est_error = []
y_ext_est_error = []
z_ext_est_error = []

for i in range(0, n_imgs):
    angle1.append(np.sqrt(C_ext[6*i, 6*i]))
    angle2.append(np.sqrt(C_ext[6*i+1, 6*i+1]))
    angle3.append(np.sqrt(C_ext[6*i+2, 6*i+2]))
    x_ext_est_error.append(np.sqrt(C_ext[6*i+3, 6*i+3]))
    y_ext_est_error.append(np.sqrt(C_ext[6*i+4, 6*i+4]))
    z_ext_est_error.append(np.sqrt(C_ext[6*i+5, 6*i+5]))
# Extract covariance for the validation dataset
estimatedVx = []
estimatedVy = []
estimatedVz = []
trueVx = []
trueVy = []
trueVz = []
# Use this to estimate all
'''
for i in range(len(coords_3d)):
    estimatedVx.append(np.sqrt(abs(cov_X_new[i * 3, i * 3])))
    estimatedVy.append(np.sqrt(abs(cov_X_new[i * 3 + 1, i * 3 + 1])))
    estimatedVz.append(np.sqrt(abs(cov_X_new[i * 3 + 2, i * 3 + 2])))
'''
# Find the validation points in sfm corresponding to 
validation_sfm = np.zeros((lidar_3d_x.shape[0], 3))
dist = []
x_center = 273480.15
y_center = 3289623.31
for idx in range(lidar_3d_x.shape[0]):
    i = int(sfm_3D_idx[idx])
    validation_sfm[idx, 0] = XYZ_3D_new[3*i] + 273000
    validation_sfm[idx, 1] = XYZ_3D_new[i * 3 + 1] + 3289000
    validation_sfm[idx, 2] = XYZ_3D_new[i * 3 + 2]
    if (abs(lidar_3d_z[idx] - validation_sfm[idx, 2]) > 3 ):
        continue
    else:
        trueVx.append(abs(lidar_3d_x[idx] - validation_sfm[idx, 0]))
        trueVy.append(abs(lidar_3d_y[idx] - validation_sfm[idx, 1]))
        trueVz.append(abs(lidar_3d_z[idx] - validation_sfm[idx, 2]))
        estimatedVx.append(np.sqrt(abs(Cx_3D[i * 3, i * 3])))
        estimatedVy.append(np.sqrt(abs(Cx_3D[i * 3 + 1, i * 3 + 1])))
        estimatedVz.append(np.sqrt(abs(Cx_3D[i * 3 + 2, i * 3 + 2])))
        x_diff = validation_sfm[idx, 0] - x_center
        y_diff = validation_sfm[idx, 1] - y_center
        dist.append(np.sqrt(x_diff ** 2 + y_diff ** 2))

XYZ_new = np.reshape(XYZ_3D_new, (len(coords_3d), 3))
np.savetxt('titan_georef_XYZ.txt', XYZ_new)
np.savetxt('titan_validation_sfm.txt', validation_sfm)

# Calculate ratio between estimated and measured
ratioX = [i / j for i, j in zip(trueVx, estimatedVx)]
ratioY = [i / j for i, j in zip(trueVy, estimatedVy)]
ratioZ = [i / j for i, j in zip(trueVz, estimatedVz)]

# calculate distance to center


fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].scatter(trueVx, estimatedVx, s = 3)
axs[0].set_title('X coordinates')


fig.suptitle('Estimated Errors vs. Measured Errors', fontsize=16)

axs[1].scatter(trueVy, estimatedVy, s = 3)
axs[1].set_title('Y coordinates')
axs[1].set_ylabel('Unsigned Estimated Error (m)')


axs[2].scatter(trueVz, estimatedVz, s = 3)
axs[2].set_title('Z coordinates')
axs[2].set_xlabel('Unsigned Measured Error (m)')



fig2, ax = plt.subplots(3, 1, constrained_layout=True)
fig2.suptitle('Measured Errors vs. Distance to the Center of Lidar Data', fontsize=16)
ax[0].scatter(dist, trueVx, s = 8)
ax[0].set_title('X direction')

ax[1].scatter(dist, trueVy, s = 8)
ax[1].set_title('Y direction')
ax[1].set_ylabel('Unsigned Measured Error (m)')

ax[2].scatter(dist, trueVz, s = 8)
ax[2].set_title('Z direction')
ax[2].set_xlabel('Distance to the Center of Lidar Data (m)')


# Read Exterior truth data
path_ground_truth = "E:/SFM/wfh/UH_campus/Data/"
exterior_truth_df = pd.read_csv(path_ground_truth + "Exterior orientation.txt",  sep = '\s+', header=None) 
exterior_truth = exterior_truth_df.to_numpy()

exterior_truth = np.concatenate((exterior_truth[0, 1:7], exterior_truth[1, 1:7], \
    exterior_truth[2, 1:7], exterior_truth[8, 1:7], exterior_truth[9, 1:7], \
        exterior_truth[10, 1:7], exterior_truth[11, 1:7], exterior_truth[12, 1:7], \
            exterior_truth[13, 1:7], exterior_truth[3, 1:7], exterior_truth[4, 1:7], \
                exterior_truth[5, 1:7], exterior_truth[6, 1:7], exterior_truth[7, 1:7]))

show_before = False

if (show_before):
    exterior_est = exterior
else:
    exterior_est = exterior + dx[0:ue].flatten()

diff_x_ext = exterior[3::6] + dx[3:ue:6].flatten() - exterior_truth[0::6] + 273000
diff_y_ext = exterior[4::6] + dx[4:ue:6].flatten() - exterior_truth[1::6] + 3289000
diff_z_ext = exterior[5::6] + dx[5:ue:6].flatten() - exterior_truth[2::6]

fig3, ax = plt.subplots(3, 1, constrained_layout=True)
fig3.suptitle('Measured Exterior Translations Error (m)', fontsize=16)
ax[0].plot(diff_x_ext)
ax[0].set_title('X direction')

ax[1].plot(diff_y_ext)
ax[1].set_title('Y direction')

ax[2].plot(diff_z_ext)
ax[2].set_title('Z direction')

print("Average of exterior X absolute translation error: ", statistics.mean(abs(diff_x_ext)))
print("Average of exterior Y absolute translation error: ", statistics.mean(abs(diff_y_ext)))
print("Average of exterior Z absolute translation error: ", statistics.mean(abs(diff_z_ext)))


# Read CC generated X, Y, Z difference 
CC_df = pd.read_csv("E:/SFM/wfh/Data/titan_images/results/titan_georef_XYZ_CC_diff_original.txt", sep = ";", header=None, skiprows=[0]) 
x_3d = CC_df[0].to_numpy()
y_3d = CC_df[1].to_numpy()
z_3d = CC_df[2].to_numpy()
cropped_ID = CC_df[3].to_numpy()

x_measured_diff = np.absolute(CC_df[6].to_numpy())
y_measured_diff = np.absolute(CC_df[7].to_numpy())
z_measured_diff = np.absolute(CC_df[4].to_numpy())
dx_est = []
dy_est = []
dz_est = []
n_img_per_obs = []
for idx in range(0, len(cropped_ID)):
    i = int(cropped_ID[idx])
    dx_est.append(np.sqrt(Cx_3D[3 * i, 3 * i]))
    dy_est.append(np.sqrt(Cx_3D[3 * i + 1, 3 * i + 1]))
    dz_est.append(np.sqrt(Cx_3D[3 * i + 2, 3 * i + 2]))
    n_img_per_obs.append(n_obs_per_img[i])

fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].scatter(x_measured_diff, dx_est, s = 3)
axs[0].set_title('X coordinates')
axs[0].set_xlabel('Unsigned Measured Error (m)')
axs[0].set_ylabel('Unsigned Estimated Error (m)')
axs[0].set_aspect('equal')

fig.suptitle('Estimated Errors vs. Measured Errors', fontsize=16)

axs[1].scatter(y_measured_diff, dy_est, s = 3)
axs[1].set_title('Y coordinates')
axs[1].set_xlabel('Unsigned Measured Error (m)')
axs[1].set_aspect('equal')


axs[2].scatter(z_measured_diff, dz_est, s = 3)
axs[2].set_title('Z coordinates')
axs[2].set_xlabel('Unsigned Measured Error (m)')
axs[2].set_aspect('equal')

# binning the results
kx = np.linspace(0, max(x_measured_diff), 100)
ky = np.linspace(0, max(y_measured_diff), 100)
kz = np.linspace(0, max(z_measured_diff), 100)
# Compute bin index
kx_idx = np.digitize(x_measured_diff, kx)
ky_idx = np.digitize(y_measured_diff, ky)
kz_idx = np.digitize(z_measured_diff, kz)


# Compute number of points for each bin
x_area = np.bincount(kx_idx)
dx_est_area = np.bincount(kx_idx)
y_area = np.bincount(ky_idx)
dy_est_area = np.bincount(ky_idx)
z_area = np.bincount(kz_idx)
dz_est_area = np.bincount(kz_idx)

# Compute mean value for each bin
x_mean = (np.bincount(kx_idx, weights = x_measured_diff)/ np.ma.masked_where(x_area ==0, x_area))
dx_est_mean = (np.bincount(kx_idx, weights = dx_est)/ np.ma.masked_where(dx_est_area ==0, dx_est_area))
y_mean = (np.bincount(ky_idx, weights = y_measured_diff)/ np.ma.masked_where(y_area ==0, y_area))
dy_est_mean = (np.bincount(ky_idx, weights = dy_est)/ np.ma.masked_where(dy_est_area ==0, dy_est_area))
z_mean = (np.bincount(kz_idx, weights = z_measured_diff)/ np.ma.masked_where(z_area ==0, z_area))
dz_est_mean = (np.bincount(kz_idx, weights = dz_est)/ np.ma.masked_where(dz_est_area ==0, dz_est_area))

plt.scatter(x_mean, dx_est_mean, s = 5)
plt.scatter(y_mean, dy_est_mean, s = 5)
plt.scatter(z_mean, dz_est_mean, s = 5)

plt.title('Average of each bin of estimated error vs. measured error (100 bins)')
plt.legend(['x', 'y', 'z'])
plt.xlabel('Mean Measured Error (m)')
plt.ylabel('Mean Estimated Error (m)')


fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].scatter(abs(diff_x_ext), x_ext_est_error, s = 3)
axs[0].set_title('X Exterior')


fig.suptitle('Estimated Exterior Orientation Errors (Translation only) vs. Measured Errors', fontsize=16)

axs[1].scatter(abs(diff_y_ext), y_ext_est_error, s = 3)
axs[1].set_title('Y Exterior')
axs[1].set_ylabel('Unsigned Estimated Error (m)')


axs[2].scatter(abs(diff_z_ext), z_ext_est_error, s = 3)
axs[2].set_title('Z Exterior')
axs[2].set_xlabel('Unsigned Measured Error (m)')


# Plot some histograms
fig, axs = plt.subplots(3, 2, constrained_layout = True)
fig.suptitle('Histograms of Measured Errors and Estimated Errors in Each Axis')
n, bins, patches = axs[0, 0].hist(x=x_measured_diff, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
axs[0, 0].grid(axis='y', alpha = 0.75)
axs[0, 0].set_xlabel('Value')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].set_title('Measured Error in X (m)')
axs[0, 0].text(0.2, 250, r'$\mu=0.108, b=0.108$')

n, bins, patches = axs[0, 1].hist(x=dx_est, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
axs[0, 1].grid(axis='y', alpha = 0.75)
axs[0, 1].set_xlabel('Value')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].set_title('Estimated Error in X (m)')
axs[0, 1].text(0.1, 200, r'$\mu=0.055, b=0.030$')

n, bins, patches = axs[1, 0].hist(x=y_measured_diff, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
axs[1, 0].grid(axis='y', alpha = 0.75)
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_title('Measured Error in Y (m)')
axs[1, 0].text(0.2, 250, r'$\mu=0.086, b=0.104$')

n, bins, patches = axs[1, 1].hist(x=dy_est, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
axs[1, 1].grid(axis='y', alpha = 0.75)
axs[1, 1].set_xlabel('Value')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].set_title('Estimated Error in Y (m)')
axs[1, 1].text(0.1, 200, r'$\mu=0.055, b=0.030$')

n, bins, patches = axs[2, 0].hist(x=z_measured_diff, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
axs[2, 0].grid(axis='y', alpha = 0.75)
axs[2, 0].set_xlabel('Value')
axs[2, 0].set_ylabel('Frequency')
axs[2, 0].set_title('Measured Error in Z (m)')
axs[2, 0].text(0.5, 150, r'$\mu=0.235, b=0.176$')

n, bins, patches = axs[2, 1].hist(x=dz_est, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
axs[2, 1].grid(axis='y', alpha = 0.75)
axs[2, 1].set_xlabel('Value')
axs[2, 1].set_ylabel('Frequency')
axs[2, 1].set_title('Estimated Error in Z (m)')
axs[2, 1].text(0.4, 200, r'$\mu=0.251, b=0.140$')


# Plot estimated errors against number of images it's observed in
fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].scatter(n_img_per_obs, dx_est, s = 3)
axs[0].set_title('X coordinates')
axs[0].set_xlabel('Number of Photos')
axs[0].set_ylabel('Unsigned Estimated Error (m)')

fig.suptitle('Estimated Errors vs. Number of Photos the Point Appeared in', fontsize=16)

axs[1].scatter(n_img_per_obs, dy_est, s = 3)
axs[1].set_title('Y coordinates')
axs[1].set_xlabel('Number of Photos')


axs[2].scatter(n_img_per_obs, dz_est, s = 3)
axs[2].set_title('Z coordinates')
axs[2].set_xlabel('Number of Photos')

# Plot measured errors
fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].scatter(n_img_per_obs, x_measured_diff, s = 3)
axs[0].set_title('X coordinates')
axs[0].set_xlabel('Number of Photos')
axs[0].set_ylabel('Unsigned Measured Error (m)')

fig.suptitle('Measured Errors vs. Number of Photos the Point Appeared in', fontsize=16)

axs[1].scatter(n_img_per_obs, y_measured_diff, s = 3)
axs[1].set_title('Y coordinates')
axs[1].set_xlabel('Number of Photos')


axs[2].scatter(n_img_per_obs, z_measured_diff, s = 3)
axs[2].set_title('Z coordinates')
axs[2].set_xlabel('Number of Photos')
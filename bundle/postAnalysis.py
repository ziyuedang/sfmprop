import numpy as np
import scipy.io as SIO
from scipy import spatial
import json
import pandas as pd
import random
import time
import sfm_IO
import matplotlib.pyplot as plt


sfm_path = "E:/SFM/wfh/Data/titan_images/downsampled_photo/reconstruction_global/"
sfm_filename = "sfm_data.json"

# read SfM data
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = sfm_IO.read_sfm(sfm_filename, sfm_path)
n_imgs = len(extrinsics)

# Gather keypoints for each image
keys_imgs = [np.empty([0, 3])] * n_imgs # This is merged observations for all images

temp = np.empty([0, 3])
for i in range(0, len(coords_3d)):
    obs = coords_3d[i]['observations']
    for j in range(0, len(obs)):
        k = obs[j]['key']
        temp_id_feat = obs[j]['value']['id_feat']
        temp_x = obs[j]['value']['x'][0]
        temp_y = obs[j]['value']['x'][1]
        temp = np.array([temp_id_feat, temp_x, temp_y])
        keys_imgs[k] = np.append(keys_imgs[k], [temp], axis = 0)

# save keypoints for each image to file
SIO.savemat('downsampled_merged_obs.mat', {'root_path': root_path, 'keys_imgs': keys_imgs, 'views_meta': views_meta})

# Load manually selected validation points from .mat file
mat_filename = 'E:/SFM/wfh/Data/titan_images/match_orig/tie_points_2D.mat'
validation_data = SIO.loadmat(mat_filename)
validation_points = validation_data['result']

for i in range(0, len(validation_points)):
    viewID = validation_points[i][0][0][0][0][0, 0] - 1
    img_validation_points_pair = validation_points[i][0][0][0][1]
    orig_img_points = img_validation_points_pair[:, 0:2]
    intensity_img_points = img_validation_points_pair[:, 2:4]

    # intensity 2D -> LIDAR 3D

def img2lidar()




'''
def nearest(lasXYZ, pt):
    distance, index = spatial.KDTree(lasXYZ).query(pt)
    return distance, index

def distXYZ(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    distX = x1 - x2
    distY = y1 - y2
    distZ = z1 - z2
    return distX, distY, distZ

def dist3D(distX, distY, distZ):
    distance = np.sqrt(distX ** 2 + distY ** 2 + distZ ** 2)
    return distance

n_imgs = 14
lasdata = SIO.loadmat('Titan.mat')
lasX = lasdata['lasX']
lasY = lasdata['lasY']
lasZ = lasdata['lasZ']

lasXYZ = np.concatenate((lasX, lasY, lasZ), axis=1)

path_sfm = 'E:/SFM/wfh/UH_campus/ALS/'
sfm_filename = 'structure_04_outlier_removed-georeg.txt'
filename = path_sfm + sfm_filename
with open(filename, 'r') as openfile:
    content = pd.read_csv(openfile, sep = '\s+', header=None)
content = content.drop([3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1)

sfm = np.delete(content.to_numpy(), np.s_[0:n_imgs], 0)
subset_idx = random.sample(range(0, sfm.shape[0]), 100)
sfm_subset = sfm[subset_idx, :]
distance = []
index = []
start = time.time()
for i in range(sfm_subset.shape[0]):
    pt = sfm[i, :]
    dist, idx = nearest(lasXYZ, pt)
    distance.append(dist)
    index.append(idx)

end = time.time()
print('Completed in: ', end-start)

las_coords = lasXYZ[index, :]
distanceX = []
distanceY =[]
distanceZ = []
for j in range(len(distance)):
    distX, distY, distZ = distXYZ(lasXYZ[index[j], :], sfm[j, :])
    distanceX.append(distX)
    distanceY.append(distY)
    distanceZ.append(distZ)
cx = SIO.loadmat('campus_scaled_georeg_cx.mat')
cov = cx['cov_unk_3D']
Vx = np.sqrt(abs(cov[0:i:3, 0:i:3]))
Vy = np.sqrt(abs(cov[1:i:3, 1:i:3]))
Vz = np.sqrt(abs(cov[2:i:3, 2:i:3]))

SIO.savemat('campus_errors.mat', {'diffX':np.array(distanceX), 'diffY':np.array(distanceY), 'diffZ':np.array(distanceZ), 'Vx': Vx, 'Vy':Vy, 'Vz':Vz})
SIO.savemat('campus_las_idx.mat', {'idx':index})
'''


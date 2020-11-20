import numpy as np
import scipy.io as SIO
from scipy import spatial
import pdal
import json
import pandas as pd
import random
import time

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
from laspy.file import File
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as SIO
from scipy.spatial import KDTree
from tqdm import tqdm
import pandas as pd
import open3d as o3d
import copy
import statistics

from open3d import visualization, registration, utility, geometry
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Load lidar data
inFile = File("E:/SFM/wfh/UH_campus/ALS/Titan_data.las", mode = 'r')
point_records = inFile.points
x = np.asarray(inFile.x)
y = np.asarray(inFile.y)
z = np.asarray(inFile.z)

# Load SfM key point point cloud
sfm_key = SIO.loadmat('campus_scaled.mat')
XYZ_key = sfm_key['XYZ_georef']
X_key = XYZ_key[0::3]
Y_key = XYZ_key[1::3]
Z_key = XYZ_key[2::3]

# Load densified SfM point cloud
sfm = o3d.io.read_point_cloud("E:/SFM/wfh/Data/campus_p_0.12_ANNL2_new/pmvs/PMVS/models/campus_mvs.ply")
sfm_dense = np.asarray(sfm.points)
XYZ = sfm_dense.flatten()
X_sfm = XYZ[0::3]
Y_sfm = XYZ[1::3]
Z_sfm = XYZ[2::3]
sfm_key_2D = np.transpose(np.stack((X_key.flatten(), Y_key.flatten())))
# Dense PC Georegistration with lidar data - parameters are calculated from cloudcompare
s_georef = 1
R_georef = np.array([[-3.695, 383.316, -5.996], [383.360, 3.668, -1.715], [-1.658, -6.012, -383.330]])
t = np.array([[46.985], [-241.142], [472.254]])
t2 = np.array([[273700], [3289700], [0]])
XYZ_ref = np.empty((len(XYZ), 1))
for i in range(len(X_sfm)):
    XYZ_temp = np.array([[XYZ[3*i]], [XYZ[3*i+1]], [XYZ[3*i+2]]])
    XYZ_ref[3*i:3*(i+1)] = s_georef * np.dot(R_georef, XYZ_temp) + t + t2

X_sfm = XYZ_ref[0::3]
Y_sfm = XYZ_ref[1::3]
Z_sfm = XYZ_ref[2::3]



sfm = np.transpose(np.stack((X_sfm.flatten(), Y_sfm.flatten())))
lidar_points = np.transpose(np.stack((x, y)))
tree = KDTree(lidar_points)

# Filter out patches with less than 30 points
# Find neighbors of sfm key point in als point cloud with a 10m radius
valid_key_index = []
results = []
for i in tqdm(range(0, len(X_key))):
    result = tree.query_ball_point(sfm_key_2D[i, :], 10)
    z_diff = z[result] - Z_key[i]
    z_sifted_index = [idx for idx in range(len(z_diff)) if abs(z_diff[idx]) < 1]
    result_new = [result[index] for index in z_sifted_index]
    if len(result_new) == 0:
        result_new = [0]
    results.append(result_new)

for i in tqdm(range(0, len(X_key))):
    result = results[i]
    if len(result) < 30:
        continue
    valid_key_index.append(i)

# Find neighbors of sfm keypoint in dense sfm point cloud with a 10 m radius
tree_sfm = KDTree(sfm)

valid_key_index_2 = []
sfm_results = []
for i in tqdm(range(0, len(X_key))):
    temp = tree_sfm.query_ball_point(sfm_key_2D[i, :], 10)
    z_diff = z[temp] - Z_key[i]
    z_sifted_index = [idx for idx in range(len(z_diff)) if abs(z_diff[idx]) < 1]
    temp_new = [temp[index] for index in z_sifted_index]
    if len(temp_new) == 0:
        temp_new = [0]
    sfm_results.append(temp_new)

for i in tqdm(range(0, len(X_key))):
    temp = sfm_results[i]
    if len(temp) < 30:
        continue       
    valid_key_index_2.append(i)

valid_index_comb = list(set(valid_key_index) & set(valid_key_index_2))
results = [results[index] for index in valid_index_comb]
sfm_results = [sfm_results[index] for index in valid_index_comb]
# update keypoints
X_key = [X_key[index] for index in valid_index_comb]
Y_key = [Y_key[index] for index in valid_index_comb]
Z_key = [Z_key[index] for index in valid_index_comb]
sfm_key_2D = np.transpose(np.stack((np.asarray(X_key).flatten(), np.asarray(Y_key).flatten())))


v_x = []
v_y = []
v_z = []
for i in tqdm(range(0, len(X_key))):
    source = o3d.geometry.PointCloud()
    index_sfm = sfm_results[i]
    source_array = np.transpose(np.stack([X_sfm[index_sfm].flatten(), Y_sfm[index_sfm].flatten(), Z_sfm[index_sfm].flatten()]))
    source.points = o3d.utility.Vector3dVector(source_array)

    target = o3d.geometry.PointCloud()
    index_lidar = results[i]
    target_array = np.transpose(np.array([x[index_lidar], y[index_lidar], z[index_lidar]]))
    target.points = o3d.utility.Vector3dVector(target_array)
    threshold = 50
    trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    """
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    """
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init,
    o3d.registration.TransformationEstimationPointToPoint(),
    o3d.registration.ICPConvergenceCriteria(max_iteration = 200))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p.transformation)

    # Apply transformation, compute rmse of Vx, Vy, Vz
    x_trans = []
    y_trans = []
    z_trans = []
    for j in range(0, np.asarray(source.points).shape[0]):
        source_trans = np.dot(reg_p2p.transformation, np.append(np.asarray(source.points)[j, :], [1]))
        x_trans.append(source_trans[0])
        y_trans.append(source_trans[1])
        z_trans.append(source_trans[2])
    reference_x = statistics.mean(np.asarray(target.points)[:, 0])
    reference_y = statistics.mean(np.asarray(target.points)[:, 1])
    reference_z = statistics.mean(np.asarray(target.points)[:, 2])
    x_trans_mean = statistics.mean(x_trans)
    y_trans_mean = statistics.mean(y_trans)
    z_trans_mean = statistics.mean(z_trans)

    v_x.append(abs(reference_x - x_trans_mean))
    v_y.append(abs(reference_y - y_trans_mean))
    v_z.append(abs(reference_z - z_trans_mean))

SIO.savemat('ICP_Errors.mat',{'v_x': v_x, 'v_y':v_y, 'v_z':v_z, 'filtered_index': valid_index_comb})

"""
clean_index = list(set.union(*map(set, results)))
result = np.array([x[clean_index],  y[clean_index], z[clean_index]])
result = np.transpose(result)
np.savetxt('Cleaned_ALS.txt', result)
"""
# Get the ICP-georegistered points from cloud compare
registered_clean_sfm_file = 'E:/SFM/wfh/UH_campus/ALS/structure_04_outlier_removed-georeg-withoutCAM-ICP.txt'
with open(registered_clean_sfm_file, 'r') as openfile:
    content = pd.read_csv(openfile, sep = '\s+', header=None)
content = content.drop([3, 4, 5, 6, 7, 8, 10, 11, 12], axis=1)
index = content.iloc[:,-1].to_list()
index = [int(i) for i in index]
reducedTree = KDTree(result)
XYZ_sfm = content.iloc[:, 0:3].to_numpy()
# Load the scaled covariance
cx = SIO.loadmat('campus_scaled_georeg_cx.mat')
cov = cx['cov_unk_3D']
covariance = cov.diagonal()
Vx = np.sqrt(abs(covariance[0::3]))
Vy = np.sqrt(abs(covariance[1::3]))
Vz = np.sqrt(abs(covariance[2::3]))
Vx = Vx[index]
Vy = Vy[index]
Vz = Vz[index]

idx = []
dist = []
for i in tqdm(range(0, len(Vx))):
    distance, idx_nearest = reducedTree.query(XYZ_sfm[i, :], 1)
    dist.append(distance)
    idx.append(idx_nearest)

threshold = np.mean(dist) + 2 * np.std(dist)
index_thresh = np.where(dist<threshold)[0]
als_index = [idx[i] for i in index_thresh]

xyz_als = result[als_index, :]
xyz_sfm_3D = XYZ_sfm[index_thresh, :]
Vx = Vx[index_thresh]
Vy = Vy[index_thresh]
Vz = Vz[index_thresh]

x_diff = abs(xyz_als[:, 0] - xyz_sfm_3D[:, 0])
y_diff = abs(xyz_als[:, 1] - xyz_sfm_3D[:, 1])
z_diff = abs(xyz_als[:, 2] - xyz_sfm_3D[:, 2])
xyz_als_refined = result[idx, :]
SIO.savemat('campus_las_sfm_final.mat', {'als_data':xyz_als_refined, 'XYZ': XYZ_sfm, 'Vx':Vx, 'Vy': Vy, 'Vz':Vz})

fig = plt.figure()

imgs_path = {'E:\SFM\wfh\Data\campus\11.JPG';'E:\SFM\wfh\Data\campus\12.JPG';...
    'E:\SFM\wfh\Data\campus\13.JPG'; 'E:\SFM\wfh\Data\campus\30.JPG'; ...
    'E:\SFM\wfh\Data\campus\31.JPG';'E:\SFM\wfh\Data\campus\48.JPG';...
    'E:\SFM\wfh\Data\campus\49.JPG';'E:\SFM\wfh\Data\campus\50.JPG';...
    'E:\SFM\wfh\Data\campus\167.JPG';'E:\SFM\wfh\Data\campus\168.JPG';...
    'E:\SFM\wfh\Data\campus\169.JPG';'E:\SFM\wfh\Data\campus\177.JPG';...
    'E:\SFM\wfh\Data\campus\178.JPG';'E:\SFM\wfh\Data\campus\179.JPG'};
% Key 0, 1, 2 - 11, 12, 13
% Key 3, 4, 5 - 167, 168, 169
% Key 6, 7, 8 - 177, 178, 179
% Key 9, 10 - 30, 31
% Key 11, 12, 13 - 48, 49, 50

%%================================================================
% Pick corresponding points
%idx_all = [idx_all, index];
i = 1;
key = [1, 2, 3, 9, 10, 11, 12, 13, 14, 4, 5, 6, 7, 8];
img = imread(imgs_path{key(i)});
X = merged_obs{i}(:,1);
Y = merged_obs{i}(:,2);
figure;
imshow(img)
hold on;
scatter(X, Y, 10, 'r', 'filled');

%==================================

%%=========================Read data==============================
load('campus_scaled.mat')
load('ICP_Errors.mat')
filtered_index = filtered_index + 1;
%==========Match 3D points from sfm to the filtered patches=========
X_sfm = XYZ_georef(1:3:end);
Y_sfm = XYZ_georef(2:3:end);
Z_sfm = XYZ_georef(3:3:end);
X_sfm_filtered = X_sfm(filtered_index);
Y_sfm_filtered = Y_sfm(filtered_index);
Z_sfm_filtered = Z_sfm(filtered_index);

cov_xyz = diag(cov_unk_3D);
Vx = sqrt(abs(cov_xyz(1:3:end)));
Vy = sqrt(abs(cov_xyz(2:3:end)));
Vz = sqrt(abs(cov_xyz(3:3:end)));

Vx_filtered = Vx(filtered_index);
Vy_filtered = Vy(filtered_index);
Vz_filtered = Vz(filtered_index);

% 2 sigma filters
icp_2_sigma = [2 * std(v_x), 2* std(v_y), 2*std(v_z)];
estimated_2_sigma = [2 * std(Vx_filtered), 2 * std(Vy_filtered), 2 * std(Vz_filtered)];
mean_icp = [mean(v_x), mean(v_y), mean(v_z)];
mean_estimated = [mean(Vx_filtered), mean(Vy_filtered), mean(Vz_filtered)];
icp_low = mean_icp - icp_2_sigma;
icp_high = mean_icp + icp_2_sigma;
estimated_low = mean_estimated - estimated_2_sigma;
estimated_high = mean_estimated + estimated_2_sigma;

for i = 1:length(Vx_filtered)
    if v_x > icp_low(1) && v_y > icp_low(2) && v_z > icp_low(3) && v
            


figure(3)
subplot(2, 2, 1);
scatter(Vx_filtered, v_x, 'filled' ,'r');
hold on;
scatter(Vy_filtered, v_y, 'filled' , 'b');
hold on;
scatter(Vz_filtered, v_z, 'filled', 'g');
title('ICP Mean Measured Error vs. Mean Estimated Error');
xlabel('Estimated error on SfM point cloud propagated from SIFT (m)');
ylabel('Difference between SfM point cloud and lidar (m)');
legend('X', 'Y', 'Z')

subplot(2, 2, 2);
scatter(Vx_filtered, v_x, 'filled' ,'r');
title('Error in X direction')

subplot(2, 2, 3);
scatter(Vy_filtered, v_y, 'filled' , 'b');
title('Error in Y direction')

subplot(2, 2, 4);
scatter(Vz_filtered, v_z, 'filled', 'g');
title('Error in Z direction')
%% 
XYZ = reshape(XYZ_georef, [3, length(XYZ_georef)/3]);
id_3D = [];
XY_ID = cell(10,1);
for i = 1:10
    img = eval(['img',num2str(i)]);
    X = merged_obs{i}(:,1);
    Y = merged_obs{i}(:,2);
    size = length(img);
    temp = [];
    for j = 1:size
        g = img(j).Position;
        D = pdist2([X Y], g);
        [~, idx] = min(D);
        temp = [temp, idx];
        XY_ID{i} = temp;
        id_3D = [id_3D, merged_obs{i}(idx, 3) + 1];
    end
end


XYZ_keypoints = XYZ(:,id_3D);

% Exclude the outlier (human error)
XYZ_keypoints(:, 2) = [];
als_data(5, :) = [];
id_3D(2) = [];
figure(2)
scatter3(XYZ_keypoints(1,:), XYZ_keypoints(2,:), XYZ_keypoints(3,:), 25, 'r','d', 'filled');
hold on;
scatter3(als_data(:,2), als_data(:,3), als_data(:,4),50,'b', 'o');
sensor_size = 53.904e-3;
pixel_size = 6e-6;
tempId = [];
for i = 1:length(id_3D)
    x_sfm = XYZ_keypoints(1, i);
    y_sfm = XYZ_keypoints(2, i);
    z_sfm = XYZ_keypoints(3, i);
    als_xy = [als_data(:, 2), als_data(:, 3)];
    dist = pdist2([x_sfm y_sfm], als_xy);
    [~, tempIdx] = min(dist);
    tempId = [tempId, tempIdx];
end
als = als_data(tempId, :);
cov_xyz = diag(cov_unk_3D);
Vx = sqrt(abs(cov_xyz(1:3:end)));
Vy = sqrt(abs(cov_xyz(2:3:end)));
Vz = sqrt(abs(cov_xyz(3:3:end)));
id_3D = 1:length(idx);
Vx = Vx(id_3D);
Vy = Vy(id_3D);
Vz = Vz(id_3D);
diffX = transpose(XYZ_keypoints(1,:) - transpose(als(:,2)));
diffY = transpose(XYZ_keypoints(2,:) - transpose(als(:,3)));
diffZ = transpose(XYZ_keypoints(3,:) - transpose(als(:,4)));

% this is running all the 3D sfm points
diffX = XYZ(:,1) - als_data(:,1);
diffY = XYZ(:,2) - als_data(:,2);
diffZ = XYZ(:,3) - als_data(:,3);

ratioX = diffX./Vx';
ratioY = diffY./Vy';
ratioZ = diffZ./Vz';
figure(3)
subplot(2, 2, 1);
scatter(Vx, abs(diffX), 'filled' ,'r');
hold on;
scatter(Vy, abs(diffY), 'filled' , 'b');
hold on;
scatter(Vz, abs(diffZ), 'filled', 'g');
title('Measured Error vs. Estimated Error');
xlabel('Estimated error on SfM point cloud propagated from SIFT (m)');
ylabel('Difference between SfM point cloud and lidar (m)');
legend('X', 'Y', 'Z')

subplot(2, 2, 2);
scatter(Vx, abs(diffX), 'filled' ,'r');
title('Error in X direction')

subplot(2, 2, 3);
scatter(Vy, abs(diffY), 'filled' , 'b');
title('Error in Y direction')

subplot(2, 2, 4);
scatter(Vz, abs(diffZ), 'filled', 'g');
title('Error in Z direction')

figure(4)
plot(ratioX, 'r')
hold on;
plot(ratioY, 'b')
hold on;
plot(ratioZ, 'g')
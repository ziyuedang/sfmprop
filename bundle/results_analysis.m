
%% Reformat covariance matrix
cov_obs = cell(length(obs_count),1);
v_x = cell(length(obs_count), 1);
v_y = cell(length(obs_count), 1);
upper_idx = cumsum(obs_count);
temp = [0, upper_idx];
lower_idx = temp(1:length(obs_count));
obs_x = obs(1:2:end);
obs_y = obs(2:2:end);
x = cell(length(obs_count), 1);
y = cell(length(obs_count), 1);
imgs_path = {'D:\Documents\Research\Data\LessData\IMG_0427.JPG';'D:\Documents\Research\Data\LessData\IMG_0428.JPG';...
    'D:\Documents\Research\Data\LessData\IMG_0429.JPG'; 'D:\Documents\Research\Data\LessData\IMG_0430.JPG'; ...
    'D:\Documents\Research\Data\LessData\IMG_0431.JPG'; 'D:\Documents\Research\Data\LessData\IMG_0432.JPG'};
residual_x = V(1:2:end);
residual_y = V(2:2:end);
for i = 1:length(cov_obs)
    cov_obs{i} = C_l_adjusted((2*lower_idx(i)+1):upper_idx(i)*2, (2*lower_idx(i)+1):upper_idx(i)*2);
    v_x{i} = residual_x((lower_idx(i)+1): upper_idx(i));
    v_y{i} = residual_y((lower_idx(i)+1): upper_idx(i));
    x{i} = obs_x((lower_idx(i)+1): upper_idx(i));
    y{i} = obs_y((lower_idx(i)+1): upper_idx(i));
    
end
number_exterior_unknowns = 36;
absVx = cellfun(@abs, v_x, 'UniformOutput', false);
absVy = cellfun(@abs, v_y,'UniformOutput', false);
maxVx = cellfun(@max, absVx);
maxVy = cellfun(@max, absVy);
stdVx = cellfun(@std, absVx);
stdVy = cellfun(@std, absVy);
midVx = cellfun(@median, absVx);
midVy = cellfun(@median, absVy);
minVx = cellfun(@min, absVx);
minVy = cellfun(@min, absVy);
meanVx = cellfun(@mean, absVx);
meanVy = cellfun(@mean, absVy);
exterior = Unk(1:number_exterior_unknowns);
X = Unk(number_exterior_unknowns+1:3:end);
Y = Unk(number_exterior_unknowns+2:3:end);
Z = Unk(number_exterior_unknowns+3:3:end);
for i = 1:length(cov_obs)
    img = imread(imgs_path{i});
    x_img = x{i};
    y_img = y{i};
    covariance = cov_obs{i};
    figure(i);
    imshow(img)
    hold on;
    for j = 1:length(x{i})
        x_coord = x_img(j);
        y_coord = y_img(j);
        cov = covariance((2*j-1):(2*j), (2*j-1):(2*j));
        plotErrorEllipse(x_coord, y_coord, cov)
    end

    

end


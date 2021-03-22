import os
import sys
import subprocess
from glob import glob
parent_dir = "E:/SFM/wfh/Data/titan_images/output"
image_dir_list = os.listdir(parent_dir)
all_nonEmptyDir = []
for i in image_dir_list:
    image_dir = os.path.join(parent_dir, str(i))
    patch_dir_list = glob(image_dir + '/*/')
    nonEmptyDir = []
    for p in patch_dir_list:
        patch_dir = os.path.join(image_dir, p)
        matchExport = os.path.join(patch_dir, 'matches\\export\\')
        currDir = os.listdir(matchExport)
        if len(currDir) == 0:
            print("This folder '%s' is empty", p)
        else:
            nonEmptyDir.append(p)
    all_nonEmptyDir.append(nonEmptyDir)
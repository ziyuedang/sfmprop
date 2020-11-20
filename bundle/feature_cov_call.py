# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:43:24 2020

@author: zdang2
"""

# Indicate the sift_cov binary directory
SIFT_COV_BIN = "D:/Documents/Research/dev/build/Release/"

# Indicate the image data directory
IMAGE_DIR = "D:/Documents/Research/Data/"

import os

# Number of images
def call_sub(SIFT_COV_BIN, IMAGE_DIR, file_name):
    file_name = os.path.join(IMAGE_DIR, file_name)
    print("Computing feature covariances for image file: " + file_name)
    cmd = "D:/SFM/covestimator/covEstimate/bin/x64/Release/sift_cov.exe" + "-i" + file_name + "-d" + IMAGE_DIR + "-show"
    os.sys(cmd)
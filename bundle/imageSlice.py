"""
Image Slicing because files are too large.
"""
import os
import image_slicer
file_dir = "D:/Documents/Research/Data/napa/"
files = os.listdir(file_dir)
filenames = [file_dir + i for i in files if i.endswith('tif')]

for file in filenames:
    image_slicer.slice(file, 36)
import numpy as np
import cv2
import cyvlfeat as vlfeat
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte
from skimage.transform import resize, rescale
from scipy import ndimage
import os
import subprocess
import scipy
import visualize
import re

def estimation(files, names, scale):
    print("starting estimation")

    O = np.load('../npy/patches.npy')
    O, W, _, _ = initialization(O)

def initialization(O):
    print("initializing")

    #used to initialize albedo, gamma, constants, and indicator mat
    print(O.shape)
    num_img = O[0].shape[1]
    num_pts = O[0].shape[0]

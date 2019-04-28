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

    O = np.load('../npy/observation.npy')
    O, W, _, _ = initialization(O)

    lbd = 1/np.sqrt(np.amin([O[0].shape[0], O[0].shape[1]]))
    rho = 1.05 ## do not change

    num_img = O[0].shape[1]
    num_pts = O[0].shape[0]

    albedo, const, gamma = [], [], []
    O_low = []

    


def initialization(O):
    print("initializing")

    #used to initialize albedo, gamma, constants, and indicator mat
    num_img = O[0].shape[1]
    num_pts = O[0].shape[0]

    gamma = []
    cons = []
    W = np.zeros((num_pts, num_img))
    v_id = []

    for ch in range(0,3):
        gamma.append(np.ones(num_img))
        cons.append(np.zeros(num_img))
        temp = np.where(O[ch] > 2/255)[0]
        v_id.append(temp)

    gamma = np.array(gamma)
    cons = np.array(cons)

    v_id_intersect = np.intersect1d(np.intersect1d(v_id[0], v_id[1]), v_id[2])
    W[v_id_intersect] = 1

    #### visualizing imgs
    # implement later

    #### log scaling img
    for ch in range(0,3):
        sm_o = O[ch]
        O[ch, v_id_intersect] = np.log(sm_o[v_id_intersect])

    print("initialization completed")
    return O, W, gamma, cons

import numpy as np
import cv2
import cyvlfeat as vlfeat
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize, rescale
from scipy import ndimage
import os
import subprocess
import scipy
import visualize
import re

def extract_patches_batch(files, names):

    sift_total = np.load('../npy/sift_total.npy')
    featsort = np.load('../npy/selected_featsort.npy')
    nneighvec = np.load('../npy/selected_nneighvec.npy')
    num_features = sift_total[0] #nfeat = num_features
    num_features_cum = sift_total[1]
    num_features_tot = sift_total[2]

    nselected = featsort.shape[0]
    nimg = num_features.shape[0]
    is_display_patch = True

    num_frames = len(files)
    nfeatinfoall = np.sum(nneighvec).astype(int)
    featinfoall = np.zeros((nfeatinfoall, 3))

    count = 0
    for i in range(0, nselected):

        featidtot = featsort[i]
        nfeat = featidtot.size
        imgid = np.zeros(nfeat)
        featid = np.zeros(nfeat)
        for j in range(0, nfeat):
            for k in range(0, nimg):
                if featidtot[j] <= num_features_cum[k+1]:
                    break

            imgid[j] = k
            featid[j] = featidtot[j]-num_features_cum[k]

            featinfoall[count, :] = [imgid[j], featid[j], i]
            count = count + 1

    featinfoind = np.argsort(featinfoall[:, 0])
    val = featinfoall[featinfoind]
    featinfoallsort = featinfoall[featinfoind, :].astype(int)

    patchesall = np.zeros((num_frames, 1))
    for k in range(0, num_frames):
        print("extracting patches " + str(k))
        img = io.imread(files[k])
        img = rescale(img, 6/23, anti_aliasing=True, multichannel=False, mode='reflect')
        k_ind = np.where(featinfoallsort[:,0] == k)
        t_ind = featinfoallsort[k_ind, 1].astype(int)
        si_f = np.load('../npy/'+names[k]+"_f.npy")
        # si_d = np.load('../npy/'+names[k]+"_d.npy")
        Sf = si_f[t_ind, :] # s_f should always flip
        x, y, s, t = Sf[:, 0], Sf[:, 1], Sf[:, 2] * 5, Sf[:, 3]

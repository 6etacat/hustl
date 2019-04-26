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

def extract_patches_batch(files, names):

    sift_total = np.load('../npy/sift_total.npy')
    featsort = np.load('../npy/selected_featsort.npy')
    nneighvec = np.load('../npy/selected_nneighvec.npy')
    num_features = sift_total[0] #nfeat = num_features
    num_features_cum = sift_total[1]
    num_features_tot = sift_total[2]

    num_selected = featsort.shape[0]
    num_img = num_features.shape[0]

    is_display_patch = True

    num_frames = len(files)
    num_featinfo_all = np.sum(nneighvec).astype(int)
    featinfo_all = np.zeros((num_featinfo_all, 3))

    count = 0
    for i in range(0, num_selected):
        featidtot = featsort[i]
        nfeat = featidtot.size
        imgid = np.zeros(nfeat)
        featid = np.zeros(nfeat)
        for j in range(0, nfeat):
            for k in range(0, num_img):
                if featidtot[j] <= num_features_cum[k+1]:
                    break
            imgid[j] = k
            featid[j] = featidtot[j]-num_features_cum[k]
            featinfo_all[count, :] = [imgid[j], featid[j], i]
            count = count + 1

    featinfoind = np.argsort(featinfo_all[:, 0])
    val = featinfo_all[featinfoind]
    featinfo_allsort = featinfo_all[featinfoind, :].astype(int)

    patches_all = []
    for k in range(0, num_frames):
        print("extracting patches batch" + str(k))
        img = io.imread(files[k])
        img = rescale(img, 6/23, anti_aliasing=True, multichannel=True, mode='reflect')
        k_ind = np.where(featinfo_allsort[:,0] == k)
        t_ind = featinfo_allsort[k_ind, 1].flatten().astype(int)
        si_f = np.load('../npy/'+names[k]+"_f.npy")
        Sf = si_f[t_ind, :] # s_f should always flip
        x, y, s, t = Sf[:, 0], Sf[:, 1], Sf[:, 2], Sf[:, 3]
        x, y, s, t = x.flatten(), y.flatten(), s.flatten(), t.flatten()
        patches_all.append(extract_patches(img, x, y, s, t))

    patches_all = np.array(patches_all)

    ###### organize patches
    # may need to initialize differently
    print("reorganizing patches collected")
    patches_collected = [[[0] for j in range(num_frames)] for i in range(num_selected)]
    count_col = np.zeros(num_selected).astype(int)
    count_img = np.zeros(num_frames).astype(int)
    for i in range(0, num_featinfo_all):
        img_id = int(featinfo_all[i, 0])
        col_id = int(featinfo_all[i, 2])

        # print("col_id: " + str(col_id))
        # print("img_id: " + str(img_id))
        # print("count_col[col_id]: " + str(count_col[col_id]))
        # print("count_img[img_id]: " + str(count_img[img_id]))

        patches_collected[col_id][count_col[col_id]] = patches_all[img_id][count_img[img_id]]
        count_col[col_id] = count_col[col_id] + 1
        count_img[img_id] = count_img[img_id] + 1

    patches_collected = np.array(patches_collected)
    print(patches_collected)
    print(patches_collected.shape)
    exit()


def extract_patches(img, x, y, s, t):
    img = img_as_float(img)
    height = img.shape[0]
    width = img.shape[1]

    num_feat = x.size
    x = np.rint(x)
    y = np.rint(y)
    x[x < 1] = 1
    x[x > width] = width
    y[y < 1] = 1
    y[y > height] = height

    s = np.rint(s)
    imgch = []
    for i in range(0, 3):
        imgch.append(img[:, :, i])
    imgch = np.array(imgch)

    patches = []
    for i in range(0, num_feat):
        print("extracting patches for num feat" + str(i))

        xx, yy, ss, tt = x[i], y[i], s[i], t[i]
        patch_size = int(ss * 2 + 1)
        side = np.arange(-ss, ss+1)
        xg, yg = np.meshgrid(side, side)
        R = np.array([[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]])
        xyg_rot = np.matmul(R, np.vstack([xg.reshape(-1).T, yg.reshape(-1).T]))
        xyg_rot[0, :] = xyg_rot[0, :] + xx
        xyg_rot[1, :] = xyg_rot[1, :] + yy

        xr = xyg_rot[0, :].reshape(1, -1).T # col vector now
        yr = xyg_rot[1, :].reshape(1, -1).T # col vector now
        xf, yf = np.floor(xr), np.floor(yr)
        xp = xr - xf
        yp = yr - yf

        patch = np.zeros((patch_size, patch_size, 3))

        v_id = np.where(np.logical_and(np.logical_and(xf>=1, xf<=width-1), np.logical_and(yf>=1, yf<=height-1)))
        xf, yf, xp, yp = xf[v_id], yf[v_id], xp[v_id], yp[v_id]

        ind1 = np.ravel_multi_index(((yf-1).astype(int), (xf-1).astype(int)), dims=(height, width))
        ind2 = np.ravel_multi_index(((yf-1).astype(int), (xf).astype(int)), dims=(height, width))
        ind3 = np.ravel_multi_index(((yf).astype(int), (xf-1).astype(int)), dims=(height, width))
        ind4 = np.ravel_multi_index(((yf).astype(int), (xf).astype(int)), dims=(height, width))

        for ch in range(0,3):
            one_one = np.multiply(xp, imgch[ch].ravel()[ind2])
            one_two = np.multiply((1-xp), imgch[ch].ravel()[ind1])
            two_one = np.multiply(xp, imgch[ch].ravel()[ind4])
            two_two = np.multiply((1-xp), imgch[ch].ravel()[ind3])
            i_arr = np.multiply((1-yp), (one_one + one_two)) + np.multiply((yp), (two_one + two_two))
            temp = np.zeros((patch_size, patch_size))
            temp.ravel()[v_id[0]] = i_arr
            patch[:, :, ch] = temp;

        plt.imshow(patch)
        plt.show()
        patches.append(img_as_ubyte(patch));

    patches = np.array(patches)
    return patches

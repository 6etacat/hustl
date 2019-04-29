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

    print("extracting pataches in batch")

    sift_total = np.load('../npy/sift_total.npy')
    featsort = np.load('../npy/selected_featsort.npy')
    nneighvec = np.load('../npy/selected_nneighvec.npy')
    num_features = sift_total[0] #nfeat = num_features
    num_features_cum = sift_total[1]
    num_features_tot = sift_total[2]

    num_selected = featsort.shape[0]
    num_img = num_features.shape[0]

    is_display_patch = False

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
        # visualize selected points
        # visualize_sift_points(img, Sf[:10, :])

        x, y, s, t = Sf[:, 0], Sf[:, 1], Sf[:, 2], Sf[:, 3]
        x, y, s, t = x.flatten(), y.flatten(), s.flatten(), t.flatten()
        patches_all.append(extract_patches(img, x, y, s, t))

    patches_all = np.array(patches_all)

    ###### organize patches ##########
    print("reorganizing patches collected")
    patches_collected = [[[0] for j in range(num_frames)] for i in range(num_selected)]
    count_col = np.zeros(num_selected).astype(int)
    count_img = np.zeros(num_frames).astype(int)
    for i in range(0, num_featinfo_all):
        img_id = int(featinfo_all[i, 0])
        col_id = int(featinfo_all[i, 2])

        patches_collected[col_id][count_col[col_id]] = patches_all[img_id][count_img[img_id]]
        count_col[col_id] = count_col[col_id] + 1
        count_img[img_id] = count_img[img_id] + 1

    patches_collected = np.array(patches_collected)

    if is_display_patch:
        d_patch_size = 30
        for i in range(0, num_frames):
            vis_patches = patches_all[i, 0:10]
            print(vis_patches.shape)
            for j in range(0, vis_patches.shape[0]):
                vis_patches[j] = resize(vis_patches[j], (d_patch_size, d_patch_size))

            rst_img = np.column_stack(vis_patches)
            plt.imshow(rst_img)
            plt.show()

        # for i in range(0, num_selected):
        #     d_patch_size = 30
        #     num_patches = nneighvec[i].astype(int)
        #     rst_img = np.zeros((d_patch_size, d_patch_size * num_patches, 3))
        #     rst_img = img_as_ubyte(rst_img)
        #     temp_patches = []
        #     for j in range(0, num_patches):
        #         patch = resize(patches_collected[i,j], (d_patch_size, d_patch_size))
        #         temp_patches.append(patch)
        #     rst_img = np.column_stack(temp_patches)
        #     # plt.imshow(rst_img)
        #     # plt.show()

    to_save = np.array([patches_collected, featinfo_all])
    np.save('../npy/patches', to_save)
    print("collected patches saved")

def extract_patches(img, x, y, s, t):
    print("extracting patches")
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
        # print("extracting patches for num feat" + str(i))

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

        # visualize patch
        # plt.imshow(patch)
        # plt.show()
        patches.append(img_as_ubyte(patch));

    patches = np.array(patches)
    return patches

def visualize_sift_points(img, f):
    # plot feature points onto the image
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(f[:, 1], f[:, 0], '+r', markersize=15)
    ax.axis((0, img.shape[1], img.shape[0], 0))
    plt.show()

def extract_color(files, names, is_augmentation, aug_ratio, patch_size):
    print("extracting colors")

    featsort = np.load('../npy/selected_featsort.npy')
    nneighvec = np.load('../npy/selected_nneighvec.npy')
    patches_npy = np.load('../npy/patches.npy')

    num_frames = len(files)
    patches_collected = patches_npy[0] # shape 4783, 2
    featinfo_all = patches_npy[1] # shape 9566, 3

    num_collections = patches_collected.shape[0]

    if is_augmentation:
        if aug_ratio > patch_size ** 2:
            aug_ratio = patch_size ** 2

        aug_id = np.round(np.random.rand(aug_ratio) * (patch_size**2 - 1)).astype(int)

        O_0 = np.zeros((num_collections * aug_ratio, num_frames))
        O_1 = np.zeros((num_collections * aug_ratio, num_frames))
        O_2 = np.zeros((num_collections * aug_ratio, num_frames))

    else:

        O_0 = np.zeros((num_collections, num_frames))
        O_1 = np.zeros((num_collections, num_frames))
        O_2 = np.zeros((num_collections, num_frames))

    f = fspecial_gauss(0.5 ,2) #5x5 gaussian kernal with sigma = 0.5

    count = 0
    for i in range(0, num_collections):
        for j in range(0, int(nneighvec[i])):
            img_id = featinfo_all[count, 0].astype(int)
            col_id = featinfo_all[count, 2].astype(int)
            count = count + 1
            patch = resize(patches_collected[i, j], (patch_size, patch_size))

            # if is_augmentation:
            if False:

                patch_r = patch[:,:,0]
                patch_g = patch[:,:,1]
                patch_b = patch[:,:,2]

                val_0 = img_as_float(patch_r.ravel()[aug_id]) #might cause problem here
                val_1 = img_as_float(patch_g.ravel()[aug_id])
                val_2 = img_as_float(patch_b.ravel()[aug_id])

                # print("val_0 shape:" + str(val_0.shape))
                if (col_id <= 0):
                    print(col_id)

                st_id = col_id * aug_ratio
                ed_id = (col_id+1) * aug_ratio

                O_0[st_id:ed_id, img_id] = val_0
                O_1[st_id:ed_id, img_id] = val_1
                O_2[st_id:ed_id, img_id] = val_2

            else :

                mean_0 = img_as_float(np.median(patch[:,:,0]))
                mean_1 = img_as_float(np.median(patch[:,:,1]))
                mean_2 = img_as_float(np.median(patch[:,:,2]))

                O_0[col_id, img_id] = mean_0
                O_1[col_id, img_id] = mean_1
                O_2[col_id, img_id] = mean_2

    O = np.array([O_0, O_1, O_2])

    np.save('../npy/observation', O)
    print('observation matrix saved')

def fspecial_gauss(s, k):

    """Function to mimic the 'fspecial' gaussian MATLAB function
       obtained from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """

    #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    probs = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)]
    kernel = np.outer(probs, probs)
    return np.round(kernel, decimals=4)

# import utils
import numpy as np
import cv2
import cyvlfeat as vlfeat
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize, rescale
from scipy import ndimage
import os

def main():
    path = '../../CVFinalProj_Data/'
    names = ['DSC_3160.JPG', 'DSC_3161.JPG']
    files = [path + name for name in names]

    #extract feature if they are not extracted already
    # if not os.path.isfile('../npy/'+ names[0] + '_f.npy'):
    extract_sift_feat(files, names)

def extract_sift_feat(files, names):
    ########## parameters ##########
    step_size = 10
    num_frames = len(files)
    is_single_scale = True
    scale = 2

    if is_single_scale:
        scale = 1

    is_remove_repetitive = False
    is_remove_boundary = True
    region_scale = 0.2

    if not is_remove_boundary:
        region_scale = 0

    ########## start of feature extraction ##########
    num_features = np.zeros((num_frames))

    for i in range(0, num_frames):
        img = io.imread(files[i])[0] #it reads the image 3 times for some reason
        img = color.rgb2gray(img)

        #downscale image
        img = rescale(img, 6/23, anti_aliasing=True, multichannel=False, mode='reflect')
        # trick for extracting less features
        # img = rescale(img, 2, anti_aliasing=True, multichannel=False, mode='reflect')
        img_h = img.shape[0]
        img_w = img.shape[1]

        if is_single_scale:
            #dsift can be faster
            f,d = vlfeat.sift.sift(img, peak_thresh=0.9, edge_thresh=30, compute_descriptor=True)
        else:
            f,d = vlfeat.sift.sift(img, compute_descriptor=True)

        #each row of f = [Y, X, Scale, Orientation]
        #f[:, 0 ] is Y
        #f[:, 1] is X

        #if is_remove_repetitive

        # remove features near boundary
        if is_remove_boundary:
            f = f[f[:, 1] > (img_w * region_scale)]
            f = f[f[:, 1] < (img_w * (1-region_scale))]
            f = f[f[:, 0] > (img_h * region_scale)]
            f = f[f[:, 0] < (img_h * (1-region_scale))]

        #plot feature points onto the image
        # fig, ax = plt.subplots()
        # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        # ax.plot(f[:, 1], f[:, 0], '+r', markersize=15)
        # ax.axis((0, img.shape[1], img.shape[0], 0))
        # plt.show()
        # exit()

        #saving the extracted features
        np.save('../npy/'+names[i]+"_f", f)
        np.save('../npy/'+names[i]+"_d", d)

        num_features[i] = f.shape[0] #record number of features

    ########### end of feature extraction for loop ###################
    num_features = np.array(num_features)
    num_features_cum = np.insert(np.cumsum(num_features), 0, 0)
    num_features_tot = num_features_cum[-1]

if __name__ == '__main__':
    main()

#plot feature points onto the image
# fig, ax = plt.subplots()
# ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
# # ax.plot(frames[:, 1], frames[:, 0], '.b', markersize=3)
# ax.plot(f[:20, 1], f[:20, 0], '+r', markersize=15)
# ax.axis((0, img.shape[1], img.shape[0], 0))
# plt.show()
# exit()

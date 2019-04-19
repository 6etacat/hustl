# import utils
import numpy as np
import cv2
import cyvlfeat as vlfeat
import matplotlib.pyplot as plts
from skimage import io, color
from skimage.transform import resize
from scipy import ndimage

def main():
    path = '../../CVFinalProj_Data/'
    files = [path + 'DSC_3160.JPG', path + 'DSC_3161.JPG']
    extract_sift_feat(files)

def extract_sift_feat(files):
    ########## parameters ##########
    step_size = 10

    num_frames = len(files)

    is_single_scale = True
    scale = 2

    if single_scale:
        scale = 1

    is_remove_repetitive = False
    is_remove_boundary = False
    region_scale = 0.05

    if not is_remove_boundary:
        region_scale = 0

    ########## feature extraction ##########
    nfeat = np.zeros((num_frames, 1))

    for i in range(0, num_frames):
        img = io.imread(files[i])[0] #it reads the image 3 times for some reason
        img = color.rgb2gray(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        print(img_w)
        print(img_h)

        # img = resize(img, (width, width), anti_aliasing=True)

        frames, descriptors = vlfeat.sift.dsift(img, fast=True, step=step_size)

        #plot feature points onto the image
        fig, ax = plt.subplots()
        ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        # ax.plot(frames[:, 1], frames[:, 0], '.b', markersize=3)
        ax.plot(frames[:, 1], frames[:, 0], '+r', markersize=15)
        ax.axis((0, img.shape[1], img.shape[0], 0))
        plt.show()

        # plt.imshow(img, cmap="gray");
        # plt.show();

if __name__ == '__main__':
    main()

# import numpy as np
# import cv2
from cyvlfeat.sift import dsift
import rawpy as rp
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import rescale
import concurrent.futures


def read_imgs(*fnames, debug=False):
    """Reads images with concurrency"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        iterator = zip(fnames, executor.map(read_img, fnames))
        imgs = []
        for fname, img in iterator:
            if debug:
                print(f"Finished reading {fname}")
            imgs.append(img)


def read_img(fname):
    """Reads image"""
    return io.imread(fname)


def display_img(img):
    """Displays imgage"""
    io.imshow(img)
    plt.show()


def extract_sift_features(img, step_size=10, boundary_pct=0.05, scale=6/23):
    """
    Extracts key points and their SIFT feature representations.

    :param Image img: An image to be analyzed
    :param int step_size: Steps for cyvlfeat dsift function
    :param float boundary_pct: Percentage of image to be seen as boundary
    :param float scale: Scale of rescaling (used to reduce computation)
    :returns:
        - **num_features** (*int*) - Number of feature points in the image
        - **f** (*array*) - Frames (keypoints) of the result
        - **d** (*array*) - Descriptor of corresponding frames
    """
    # make sure image is grayscale
    img = color.rgb2gray(img)
    # downscale image to extract less features
    img = rescale(img, scale=scale, anti_aliasing=True, multichannel=False,
                  mode='reflect')

    img_h, img_w = img.shape[0], img.shape[1]

    f, d = dsift(img, step=step_size, fast=True)

    # remove features near boundary
    if boundary_pct > 0:
        f = f[f[:, 1] > (img_w * boundary_pct)]
        f = f[f[:, 1] < (img_w * (1-boundary_pct))]
        f = f[f[:, 0] > (img_h * boundary_pct)]
        f = f[f[:, 0] < (img_h * (1-boundary_pct))]

    num_features = len(f)

    return num_features, f, d


# def match_features(*imgs):
#     for i in range(len(imgs)):
#         for j in range

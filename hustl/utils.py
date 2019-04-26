import numpy as np
import cv2
import cyvlfeat as vlfeat
from cyvlfeat.sift import sift
import rawpy as rp
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize, rescale
from scipy import ndimage


def extract_sift_features(img, step_size=10, single_scale=True,
                          region_scale=0.05, scale=2):
    # make sure image is grayscale
    img = color.rgb2gray(img)
    # downscale image to extract less features
    img = rescale(img, scale=6 / 23, anti_aliasing=True, multichannel=False,
                  mode='reflect')

    img_h, img_w = img.shape[0, 1]

    if single_scale: # TODO: What does this mean exactly? How is scale used?
        # sift can be faster TODO: What's this comment about?
        f, d = sift(img, peak_thresh=0.9, edge_thresh=30,
                    compute_descriptor=True)
    else:
        f, d = sift(img, compute_descriptor=True)

    # remove features near boundary
    if region_scale > 0:
        f = f[f[:, 1] > (img_w * region_scale)]
        f = f[f[:, 1] < (img_w * (1-region_scale))]
        f = f[f[:, 0] > (img_h * region_scale)]
        f = f[f[:, 0] < (img_h * (1-region_scale))]

    num_features = len(f)

    return num_features, f, d

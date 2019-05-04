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
                          region_scale=0.05, down_scale=6/23):
    # make sure image is grayscale
    img = color.rgb2gray(img)
    # downscale image to extract less features
    img = rescale(img, scale=down_scale, anti_aliasing=True, multichannel=False,
                  mode='reflect')

    # img_h, img_w = img.shape[0, 1]
    img_h, img_w = img.shape[0], img.shape[1]

    if single_scale: # TODO: What does this mean exactly? How is scale used?
        # sift can be faster TODO: What's this comment about?
        f, d = sift(img, peak_thresh=0.9, edge_thresh=30,
                    compute_descriptor=True)
    else:
        f, d = sift(img, compute_descriptor=True)

    # remove features and descriptors near boundary
    if region_scale > 0:
        left_mask = f[:, 1] > (img_w * region_scale) # remove left boarder
        f, d = f[left_mask], d[left_mask]
        right_mask = f[:, 1] < (img_w * (1-region_scale)) # remove right boarder
        f, d = f[right_mask], d[right_mask]
        bottom_mask = f[:, 0] > (img_h * region_scale) # remove bottown boarder
        f, d = f[bottom_mask], d[bottom_mask]
        top_mask = f[:, 0] < (img_h * (1-region_scale)) # remove top boarder
        f, d = f[top_mask], d[top_mask]
        

    num_features = len(f)

    return num_features, f, d
 
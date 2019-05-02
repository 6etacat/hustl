# import numpy as np
import cv2
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
    return imgs


def read_img(fname):
    """Reads image"""
    return io.imread(fname)


def display_img(img):
    """Displays imgage"""
    io.imshow(img)
    plt.show()


def extract_sift_features(img, step_size=10, boundary_pct=0.05, scale=0.26):
    """
    Extracts key points and their SIFT feature representations.

    Finds key points in the images and save them as frames ``f``, then compute
    the SIFT descriptor for these key points and save them as descriptors ``d``
    . Finally, compute the number of key points in the image and save it as
    ``num_features``.

    Parameters
    ----------
        img: Image
            An image to be analyzed
        step_size: int
            Steps for cyvlfeat dsift function
        boundary_pct: float
            Percentage of image to be seen as boundary
        scale: float
            Scale of rescaling (used to reduce computation)

    Returns
    -------
        num_features: int
            Number of feature points in the image
        fd: Set(f, d)
            - **f** (numpy.ndarray[float]) - Frames (key points) of the result
            - **d** (numpy.ndarray[uint8]) - Descriptor of corresponding frames
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
        in_boundary = ((f[:, 1] > (img_w * boundary_pct)) *
                      (f[:, 1] < (img_w * (1 - boundary_pct))) *
                      (f[:, 0] > (img_h * boundary_pct)) *
                      (f[:, 0] < (img_h * (1-boundary_pct))))
        f = f[in_boundary]
        d = d[in_boundary]

    assert len(f) == len(d)
    num_features = len(f)

    return num_features, (f, d)


def match_features(*fd, gpu=False):
    """
    Match features in multiple images

    TODO: Detailed description

    Parameters
    ----------
        *fd: Sequence[Set(f, d)]
            - **f** (numpy.ndarray[float]) - Frames (key points) of the image
            - **d** (numpy.ndarray[uint8]) - Descriptor of corresponding frames

        gpu: Bool
            Whether to use GPU for calculation

    Returns
    -------
        num_features: int
            Number of feature points in the image
    """
    if gpu:
        matcher = None  # TODO: Write GPU matcher
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    base_d = fd[0][1]
    for i in range(1, len(fd)):
        matches = matcher.match(base_d, fd[i][1])
    # FIXME: Not finished
    return sorted(matches, key=lambda x: x.distance)[:5]

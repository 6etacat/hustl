import numpy as np
from cyvlfeat.sift import sift
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

def extract_sift_features(img, peak_thresh=0.9, edge_thresh=30, boundary_pct=0.05, scale=0.26, num_keypoints=200):
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

    f, d = sift(img, peak_thresh=peak_thresh, edge_thresh=edge_thresh, compute_descriptor=True)

    # remove features near boundary
    if boundary_pct > 0:
        in_boundary = ((f[:, 1] > (img_w * boundary_pct)) *
                       (f[:, 1] < (img_w * (1 - boundary_pct))) *
                       (f[:, 0] > (img_h * boundary_pct)) *
                       (f[:, 0] < (img_h * (1-boundary_pct))))
        f = f[in_boundary]
        d = d[in_boundary]

    num_keypoints = min(num_keypoints, f.shape[0])
    keep_idx = np.random.permutation(f.shape[0])[:num_keypoints]
    f = f[keep_idx]
    d = d[keep_idx]

    assert len(f) == len(d)
    num_features = len(f)

    return num_features, (f, d)

def match_features(*fd, num_matches=80, gpu=False):
    """
    Match features in multiple images

    TODO: Detailed description

    Parameters
    ----------
        *fd: Sequence[Set(f, d)]
            - **f** (numpy.ndarray[float]) - Frames (key points) of the image
            - **d** (numpy.ndarray[uint8]) - Descriptor of corresponding frames

        num_matches: int
            Number of matches to be extracted

        gpu: Bool
            Whether to use GPU for calculation

    Returns
    -------
        best_matches: Sequence(numpy.ndarray[float])
            The frame (key points) of best matches among the given images in
            descending order
    """
    if gpu:
        matcher = None  # TODO: Write GPU matcher
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    d = fd[0][1]
    matches = []
    for i in range(0, len(fd) - 1):
        match = matcher.match(fd[i][1], fd[i + 1][1])
        matches.append(match)
    best_matches = _find_best_matches(matches, num_matches)
    return best_matches


def _find_best_matches(matches, num_matches):
    """Helper to arrange matches from best to worst and take top matches"""
    base = []
    for m in matches[0]:
        if _recurs_exist_in_all(matches, m):
            base.append(_recurs_fetch_matches(matches, m))
    best_matches = _sort_matches(base)
    return best_matches


def _recurs_fetch_matches(matches, m):
    if len(matches) == 0:
        return []
    


def _sort_matches(base):
    """Helper to sort chained matches"""
    assert len(base) > 0
    transform = []
    for i in range(len(base[0])):
        transform.append([l[i] for l in base])
    transform = sorted(transform, lambda x: np.sum([m.distance for m in x]))
    base = []
    for j in range(len(transform[0])):
        base.append([l[j] for j in transform])
    return base

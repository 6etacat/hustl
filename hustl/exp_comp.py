# import utils
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
import exp_comp_2
import exp_comp_3

def main():
        """
        This is the main function of the second stage of the pipeline (Color Consistency
        Across Frames). You can set the path to input images and hyperparameters here. This function
        calls other functions to perform color adjustments, including functions from exp_comp_2.py and
        exp_comp_3.py. The final output of this pipeline is a series of image, each corresponding to one input.

        """
    path = '../../cv_img/DSC_3188_'
    names = ["%d" % number for number in np.arange(114, 120)]
    suffix = '_res.jpg'
    files = [path + name + suffix for name in names]


    #### parameters ####
    is_hard_refresh = False # whether to run every function in the pipeline again or not.

    downscale_factor = 1 # factor for downscaling images

    is_augmentation = True # for function extract_color()
    aug_ratio = 4 # for function extract_color()
    patch_size = 30 # for function extract_color()

    scale = 10 # for function estimation()

    num_feat_kept = 1000 # number of extracted SIFT features to keep
    #### end parameters ####


    # Here begins the calling of various function. Each function' results are
    # stored as .npy or .npz. By default, each function is called once given
    # the same set of inputs to save computation time.
    if is_hard_refresh or (not os.path.isfile('../npy/sift_total.npy')):
        extract_sift_feat(files, names, num_feat_kept, downscale_factor)

    if is_hard_refresh or (not os.path.isfile('../npy/sift_match.npy')):
        match_sift_feat(files, names, downscale_factor)

    if is_hard_refresh or (not os.path.isfile('../npy/selected_featsort.npy')):
        find_clique(files, names, 2)

    if is_hard_refresh or (not os.path.isfile('../npy/patches.npy')):
        exp_comp_2.extract_patches_batch(files, names, downscale_factor)

    if is_hard_refresh or (not os.path.isfile('../npy/observation.npy')):
        exp_comp_2.extract_color(files, names, is_augmentation, aug_ratio, patch_size)

    if is_hard_refresh or (not os.path.isfile('../npy/estimation.npz')):
        exp_comp_3.estimation(files, names, scale)

    exp_comp_3.apply(files, names, downscale_factor) # applies color adjustments

def extract_sift_feat(files, names, num_feat_kept, downscale_factor):
    """
    Extracts SIFT features from each input images. Only num_feat_kept of
    all extracted features for one image are kept.

    Parameters
    ----------
        files: list of strings
            path to input images
        names: list of strings
            names of the input images
        num_feat_kept: int
            number of SIFT features to be kept and stored
        downscale_factor: float
            Factor of downscaling (used to reduce computation)

    Saves
    -------
        For each image:
            names[i]_f.npy: matrix with each row being a SIFT feature point ([Y, X, Scale, Orientation])
            names[i]_d.npy: matrix with each row being the corresponding SIFT descriptor (1x128)
        Overall:
            sift_total.npy: an array containing num_features, num_features_cum, and num_features_tot

    """

    print("-----extracting SIFT features-----")
    ########## parameters ##########
    step_size = 10
    num_frames = len(files)
    is_single_scale = True
    scale = 2

    if is_single_scale:
        scale = 1

    is_remove_boundary = False
    region_scale = 0.05

    if not is_remove_boundary:
        region_scale = 0

    ########## start of feature extraction ##########
    num_features = np.zeros((num_frames))

    for i in range(0, num_frames):
        img = io.imread(files[i]) # reads in image and greyscale
        img = color.rgb2gray(img)

        #downscale image
        img = rescale(img, downscale_factor, anti_aliasing=True, multichannel=False, mode='reflect')
        img_h = img.shape[0]
        img_w = img.shape[1]

        if is_single_scale:
            f,d = vlfeat.sift.sift(img, peak_thresh=0.98, edge_thresh=5, compute_descriptor=True)
        else:
            f,d = vlfeat.sift.sift(img, compute_descriptor=True)
            #each row of f = [Y, X, Scale, Orientation], f[:, 0 ] is Y, f[:, 1] is X

        print("Number of features extracted: " + str(f.shape[0]))

        # randomly select num_feat_kept features to keep to reduce the amount of computation
        num_feat_kept = min(num_feat_kept, f.shape[0])
        keep_idx = np.random.permutation(f.shape[0])[:num_feat_kept]
        f = f[keep_idx]
        d = d[keep_idx]

        print("Number of features kept: " + str(f.shape[0]))

        # remove features near boundary
        if is_remove_boundary:
            f = f[f[:, 1] > (img_w * region_scale)]
            f = f[f[:, 1] < (img_w * (1-region_scale))]
            f = f[f[:, 0] > (img_h * region_scale)]
            f = f[f[:, 0] < (img_h * (1-region_scale))]

        #saving the extracted features
        np.save('../npy/'+names[i]+"_f", f)
        np.save('../npy/'+names[i]+"_d", d)

        num_features[i] = f.shape[0] #record number of features

    ########### end of feature extraction for loop ###################

    # saving results to file
    num_features = np.array(num_features)
    num_features_cum = np.insert(np.cumsum(num_features), 0, 0)
    num_features_tot = num_features_cum[-1]

    to_save = np.array([num_features, num_features_cum, num_features_tot])
    np.save('../npy/sift_total', to_save)
    print("Completed extracting SIFT features")

def match_sift_feat(files, names, downscale_factor):
    """
    Matches SIFT features extracted for every image pair bi-directionally. Ensures
    all matches are unique.

    Parameters
    ----------
        files: list of strings
            path to input images
        names: list of strings
            names of the input images
        downscale_factor: float
            Factor of downscaling (used to reduce computation)

    Saves
    -------
        sift_match.npy, containing:
            M: a matrix with each rowing storing the match pairs' indicies and confidence score
            F: a matrix with each rowing the corresponding SIFT features
            num_matches: int. Total number of matches

    """

    print("-----matching SIFT features-----")

    num_frames = len(files)
    sift_total = np.load('../npy/sift_total.npy') #load previous result
    num_features = sift_total[0]

    is_vis_match = False #to see the matches or not

    # M for matches, F for features
    M = [[[0] for i in range(num_frames)] for j in range(num_frames)]
    F = [[[0] for i in range(num_frames)] for j in range(num_frames)]

    num_matches = 0;

    #for every image pair
    for i in range(0, num_frames):
        for j in range(i+1, num_frames):

            #load up feature points and descriptors.
            si_f = np.load('../npy/'+names[i]+"_f.npy")
            si_d = np.load('../npy/'+names[i]+"_d.npy")
            sj_f = np.load('../npy/'+names[j]+"_f.npy")
            sj_d = np.load('../npy/'+names[j]+"_d.npy")

            ############ start bi-directional matching #############
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            ##### match - direction 1
            cv2_matches1 = bf.match(si_d, sj_d)
            matches1 = convert_matches(cv2_matches1)  #convert matches from DMatches to arrays of [queryIdx, trainIdx]
            scores1 = extract_dist_score(cv2_matches1)

            ##### match - direction 2
            cv2_matches2 = bf.match(sj_d, si_d) #note the change in j and i
            matches2 = convert_matches(cv2_matches2) #convert matches from DMatches to arrays of [queryIdx, trainIdx]
            scores2 = extract_dist_score(cv2_matches2)
            ############ end bi-directional matching #############

            ######## find the intersection of two matches
            _, mid1, mid2 = np.intersect1d(matches1[:, 0], matches2[:, 1], return_indices=True)

            if(mid1.shape[0] == 0):
                print('no match intersects')
                continue

            matches = matches1[mid1, :]
            scores = (scores1[mid1] + scores2[mid2]) / 2

            ######## ensure unique matching pairs
            mi, mj = matches[:, 0], matches[:, 1]
            s = scores
            n = mi.shape[0]

            unique_mi, unique_mj = np.unique(mi), np.unique(mj)
            num_mi, num_mj = mi.shape[0], mj.shape[0]
            num_unique_mi, num_unique_mj = unique_mi.shape[0], unique_mj.shape[0]

            if (num_mi == num_unique_mi) and (num_mj == num_unique_mj):
                print("all matches are unique")
                num_matches = num_matches + s.shape[0]
                fi, fj = si_f[mi, :], sj_f[mj, :]
                di, dj = si_d[mi, :], sj_d[mj, :]
                M[i][j] = np.array([mi, mj, s])
                F[i][j] = np.array([fi, fj])

            else:
                # if matches are not unique, resolve by creating a sparse matrix
                print("dealing with non-unique matches")
                S_sparse = scipy.sparse.csr_matrix((s, (mi, mj)), shape=(num_features[i].astype(int), num_features[j].astype(int)))
                S = S_sparse.todense()
                for p in range(0, num_features[i].astype(int)):
                    if (S_sparse[p, :].nnz > 1):
                        Ss = S_sparse[p, :]
                        idx = np.argmax(Ss)
                        val = Ss[idx]
                        S_sparse[p, :] = 0;
                        S_sparse[p, idx] = val

                for p in range(0, num_features[j].astype(int)):
                    if (S_sparse[:, p].nnz > 1):
                        Ss = S_sparse[:, p]
                        idx = np.argmax(Ss)
                        val = Ss[idx]
                        S_sparse[:, p] = 0;
                        S_sparse[idx, p] = val

                mi, mj = np.where(S != 0)
                _,_,s_idx = scipy.sparse.find(S_sparse != 0)
                s = s[s_idx]
                num_matches = num_matches + s.shape[0]
                fi, fj = si_f[mi, :], sj_f[mj, :]
                di, dj = si_d[mi, :], sj_d[mj, :]
                M[i][j] = np.array([mi, mj, s])
                F[i][j] = np.array([fi, fj])

            ############ visualize matches #############
            if is_vis_match:
                print("visualizing matches")
                img1 = color.rgb2gray(io.imread(files[i]))
                img2 = color.rgb2gray(io.imread(files[j]))
                img1 = rescale(img1, downscale_factor, anti_aliasing=True, multichannel=False, mode='reflect')
                img2 = rescale(img2, downscale_factor, anti_aliasing=True, multichannel=False, mode='reflect')
                matches[:, 0] = mi
                matches[:, 1] = mj
                matches = matches[np.random.permutation(matches.shape[0])][:50]
                visualize.show_correspondences(img1, img2, si_f[:, 1], si_f[:, 0], sj_f[:, 1], sj_f[:, 0], matches, mode='arrows')

    # saving results to file
    M, F = np.array(M), np.array(F)
    to_save = np.array([M, F, num_matches])
    np.save('../npy/sift_match', to_save)
    print("Completed matching SIFT features")

#extract match indicies for visualization purposes
def convert_matches(matches):
    """Helper to extract indicies from OpenCV's DMatch objects"""

    xy_matches = []
    for match in matches:
        xy_matches.append([match.queryIdx, match.trainIdx])
    return np.array(xy_matches)

def extract_dist_score(matches):
    """Helper to extract confidence scores from OpenCV's DMatch objects"""

    scores = []
    for match in matches:
        scores.append(match.distance)
    return np.array(scores)

def find_clique(files, names, num_clique):
    """
    Finds maximal cliques with size greater than or equal to num_clique from a graph
    G = {V, E} where V are all the SIFT features and E are all the matches. Calls
    a C program called MACE to perform the actual clique search.

    Parameters
    ----------
        files: list of strings
            path to input images
        names: list of strings
            names of the input images
        num_clique: int
            Size of the smallest maximal clique to find

    Saves
    -------
        selected_featsort.npy:
            a matrix containing SIFT features that are part of the maximal cliques, sorted by the
            sizes of the cliques

        selected_nneighvec.npy:
            a matrix containing the sizes of each maximal clique.
    """

    print("-----finding cliques-----")

    sift_total = np.load('../npy/sift_total.npy')
    sift_match = np.load('../npy/sift_match.npy')
    num_matches = sift_match[2]
    M = sift_match[0]
    F = sift_match[1]

    num_features = sift_total[0]
    num_features_cum = sift_total[1]
    num_features_tot = sift_total[2]

    num_frames = len(files)

    ############### constructing similarity matrtix S ##########################
    Si, Sj, Ss = np.zeros(num_matches), np.zeros(num_matches), np.ones(num_matches)
    idx = 0

    # for every image pair
    for i in range(0, num_frames):
        for j in range(i+1, num_frames):
             nn_matches = M[i, j].shape[1]
             ms = M[i, j]
             if ms.shape[1] == 0:
                 continue

             mi = ms[0] + num_features_cum[i]
             mj = ms[1] + num_features_cum[j]
             s = ms[2]

             Si[idx:idx + nn_matches] = mi
             Sj[idx:idx + nn_matches] = mj
             Ss[idx:idx + nn_matches] = s
             idx = idx + nn_matches

    Si, Sj, Ss = Si.T, Sj.T, Ss.T
    ############### end constructing similarity matrtix S ##########################

    # output graph G
    mat = [np.vstack([Si, Sj]), np.vstack([Sj,Si])]
    mat = np.array(mat).T
    with open('../features/match.grh','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%d', delimiter=',')

    ########## running c program to find maximal clique
    min_clique_num = min(num_frames, num_clique)
    bat_path = r'C:\Users\majia\Desktop\CS1430\HUSTL\bin\mace.exe'
    arg1, arg2 = 'MqVe', '-l'
    input = r'C:\Users\majia\Desktop\CS1430\HUSTL\features\match.grh'
    output = r'C:\Users\majia\Desktop\CS1430\HUSTL\features\match_maximal_clique.grh'
    subprocess.call([bat_path, arg1, arg2, str(min_clique_num), input, output])

    # read in the output and clean up
    _output = "../features/match_maximal_clique.grh"
    text = []
    with open(_output, 'r') as f:
        for line in f:
            text.append(np.fromstring(line, dtype=float, sep=" "))

    text = np.array(text)
    nmatch = text.shape[0]
    sizes = np.zeros(nmatch)
    for i in range(0, nmatch):
        sizes[i] = text[i].shape[0]

    # sort the SIFT features by sizes of cliques
    sorted_idx = np.argsort(-sizes, kind='mergesort') #descending order
    sizes = sizes[sorted_idx]
    nneighvec = sizes
    featsort = np.copy(text)
    for i in range(0, nmatch):
        id = sorted_idx[i]
        featsort[i] = text[id]

    # saving results to file
    np.save('../npy/selected_featsort', featsort)
    np.save('../npy/selected_nneighvec', nneighvec)
    print("completed cliques")

if __name__ == '__main__':
    main()

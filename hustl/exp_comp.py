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
    path = '../../CVFinalProj_Data/'
    # names = ['DSC_3160.JPG', 'DSC_3161.JPG', 'DSC_3162.JPG']
    # names = ['DSC_3161.JPG', 'DSC_3162.JPG']
    names = ['DSC_3115.JPG', 'DSC_3116.JPG', 'DSC_3117.JPG', 'DSC_3118.JPG']
    # names = ['DSC_3115.JPG', 'DSC_3116.JPG']

    files = [path + name for name in names]

    is_hard_refresh = False

    #### parameters ####
    is_augmentation = True # for extract_color
    aug_ratio = 5 # for extract_color
    patch_size = 30 # for extract_color

    scale = 10 # for estimation

    #extract feature if they are not extracted already
    if is_hard_refresh or (not os.path.isfile('../npy/sift_total.npy')):
        extract_sift_feat(files, names)

    if is_hard_refresh or (not os.path.isfile('../npy/sift_match.npy')):
        match_sift_feat(files, names)

    if is_hard_refresh or (not os.path.isfile('../npy/selected_featsort.npy')):
        find_clique(files, names, 2)

    if is_hard_refresh or (not os.path.isfile('../npy/patches.npy')):
        exp_comp_2.extract_patches_batch(files, names)

    if is_hard_refresh or (not os.path.isfile('../npy/observation.npy')):
        exp_comp_2.extract_color(files, names, is_augmentation, aug_ratio, patch_size)

    if is_hard_refresh or (not os.path.isfile('../npy/estimation.npz')):
        exp_comp_3.estimation(files, names, scale)

    exp_comp_3.apply(files, names)

def extract_sift_feat(files, names):

    print("extracting SIFT features")
    ########## parameters ##########
    step_size = 10
    num_frames = len(files)
    is_single_scale = True
    scale = 2

    if is_single_scale:
        scale = 1

    is_remove_repetitive = False
    is_remove_boundary = False
    region_scale = 0.05

    if not is_remove_boundary:
        region_scale = 0

    ########## start of feature extraction ##########
    num_features = np.zeros((num_frames))

    for i in range(0, num_frames):
        img = io.imread(files[i]) #it reads the image 3 times for some reason
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

        #each row of f = [Y, X, Scale, Orientation], f[:, 0 ] is Y, f[:, 1] is X

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

        #saving the extracted features
        np.save('../npy/'+names[i]+"_f", f)
        np.save('../npy/'+names[i]+"_d", d)

        num_features[i] = f.shape[0] #record number of features

    ########### end of feature extraction for loop ###################
    num_features = np.array(num_features)
    num_features_cum = np.insert(np.cumsum(num_features), 0, 0)
    num_features_tot = num_features_cum[-1]

    to_save = np.array([num_features, num_features_cum, num_features_tot])
    np.save('../npy/sift_total', to_save)
    print("Completed extracting SIFT features")

def match_sift_feat(files, names):

    print("matching SIFT features")

    num_frames = len(files)
    sift_total = np.load('../npy/sift_total.npy')
    num_features = sift_total[0]

    is_vis_match = False #to see the matches or not

    # M for matches, F for features
    M = [[[0] for i in range(num_frames)] for j in range(num_frames)]
    F = [[[0] for i in range(num_frames)] for j in range(num_frames)]

    num_matches = 0;
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
            # cv2_matches1 = sorted(cv2_matches1, key = lambda x:x.distance) #sorted the matches based on score
            matches1 = convert_matches(cv2_matches1)  #convert matches from DMatches to arrays of [queryIdx, trainIdx]
            scores1 = extract_dist_score(cv2_matches1)

            ##### match - direction 2
            cv2_matches2 = bf.match(sj_d, si_d) #note the change in j and i
            # cv2_matches2 = sorted(cv2_matches2, key = lambda x:x.distance) #sorted the matches based on score
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
                img1 = rescale(img1, 6/23, anti_aliasing=True, multichannel=False, mode='reflect')
                img2 = rescale(img2, 6/23, anti_aliasing=True, multichannel=False, mode='reflect')
                matches[:, 0] = mi
                matches[:, 1] = mj
                matches = matches[np.random.permutation(matches.shape[0])][:50]
                visualize.show_correspondences(img1, img2, si_f[:, 1], si_f[:, 0], sj_f[:, 1], sj_f[:, 0], matches, mode='arrows')

    M, F = np.array(M), np.array(F)
    to_save = np.array([M, F, num_matches])
    np.save('../npy/sift_match', to_save)
    print("Completed matching SIFT features")

#extract match indicies for visualization purposes
def convert_matches(matches):
    xy_matches = []
    for match in matches:
        xy_matches.append([match.queryIdx, match.trainIdx])
    return np.array(xy_matches)

def extract_dist_score(matches):
    scores = []
    for match in matches:
        scores.append(match.distance)
    return np.array(scores)

def find_clique(files, names, num_clique):

    print("finding cliques")

    sift_total = np.load('../npy/sift_total.npy')
    sift_match = np.load('../npy/sift_match.npy')
    num_matches = sift_match[2]
    M = sift_match[0]
    F = sift_match[1]

    num_features = sift_total[0]
    num_features_cum = sift_total[1]
    num_features_tot = sift_total[2]

    num_frames = len(files)
    # this part is ok - dmii

    ## similarity matrtix S
    Si, Sj, Ss = np.zeros(num_matches), np.zeros(num_matches), np.ones(num_matches)
    idx = 0

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

    ####### read in the output
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

    sorted_idx = np.argsort(-sizes, kind='mergesort') #descending order
    sizes = sizes[sorted_idx]

    nneighvec = sizes
    featsort = np.copy(text)
    for i in range(0, nmatch):
        id = sorted_idx[i]
        featsort[i] = text[id]

    np.save('../npy/selected_featsort', featsort)
    np.save('../npy/selected_nneighvec', nneighvec)
    print("completed cliques")

    #try to use regex, do not work
    # text = re.findall('[^\n]*', text)
    #
    # nmatch = len(text)
    # sizes = np.zeros((1, nmatch))
    # temp_arr = np.zeros((1, nmatch))
    #
    # for i in range(0, nmatch):
    #     temp = text[i]
    #     print(temp)
    #     temp2 = re.findall('[\S]*', temp)
    #     print(temp2[0])
    #     nelement = len(temp2)
    #     arr = np.zeros((1, nelement))
    #     for j in range(0, nelement):
    #         arr[j] = float(temp2[j])
    #         print("???")
    #         print(arr[j])


# def convert_to_KeyPoints(f):
#     keypoints = []
#     for point in f:
#         keypoints.append(cv2.KeyPoint(point[0], point[1], point[2], point[3]))
#
#     return np.array(keypoints)

if __name__ == '__main__':
    main()

#plot feature points onto the image
# fig, ax = plt.subplots()
# ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
# # ax.plot(frames[:, 1], frames[:, 0], '.b', markersize=3)
# ax.plot(f[:20, 1], f[:20, 0], '+r', markersize=15)
# ax.axis((0, img.shape[1], img.shape[0], 0))
# plt.show()

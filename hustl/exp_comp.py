# import utils
import numpy as np
import cv2
import cyvlfeat as vlfeat
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize, rescale
from scipy import ndimage
import os
import scipy
import visualize

def main():
    path = '../../CVFinalProj_Data/'
    names = ['DSC_3160.JPG', 'DSC_3161.JPG']
    files = [path + name for name in names]

    #extract feature if they are not extracted already
    if not os.path.isfile('../npy/sift_total.npy'):
        extract_sift_feat(files, names)

    match_sift_feat(files, names)


def extract_sift_feat(files, names):
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
        # exit()

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

def match_sift_feat(files, names):
    num_frames = len(files)
    sift_total = np.load('../npy/sift_total.npy')
    num_features = sift_total[0]

    is_vis_match = True #to see the matches or not

    # M for matches, F for features
    M = []
    F = []

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
                M.append([mi, mj, s])
                F.append([fi, fj])

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
                M.append([mi, mj, s])
                F.append([fi, fj])

            ############ visualize matches #############
            if is_vis_match:
                img1 = color.rgb2gray(io.imread(files[i])[0])
                img2 = color.rgb2gray(io.imread(files[j])[0])
                img1 = rescale(img1, 6/23, anti_aliasing=True, multichannel=False, mode='reflect')
                img2 = rescale(img2, 6/23, anti_aliasing=True, multichannel=False, mode='reflect')
                matches[:, 0] = mi
                matches[:, 1] = mj
                matches = matches[np.random.permutation(matches.shape[0])][:50]
                visualize.show_correspondences(img1, img2, si_f[:, 1], si_f[:, 0], sj_f[:, 1], sj_f[:, 0], matches, mode='arrows')

    to_save = np.array([M, F, num_matches])
    np.save('../npy/sift_match', to_save)

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
# exit()

import hustl.utils
import hustl.hyperparameters as hp
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skvideo.io
import skimage.io
from math import inf
from numpy.linalg import norm
import os

def main():
    path = '../../CVFinalProj_Data/'
    name = 'DSC_3188'
    format = '.mov'
    video = skvideo.io.vread(path+name+format)
    frames = np.asarray(video)
    np.save('../npy/'+'frames.npy',frames)
    num_frames = frames.shape[0]
    if not os.path.isfile('../npy/'+name+'_f.npy'):
        extract_sift_features(name, frames, num_frames)
        print("----sift_features_extracted----")
    if not os.path.isfile('../npy/'+name+'_p.npy'):
        select_optimal_frames(name, frames, num_frames)
        print("----optimal_frames_selected----")
    if not os.path.isfile(path+name+'_optimal_hyperlapse.mp4'):
        generate_hyperlapse_video(name, frames, path, "optimal")
        print("----optimal_hyperlapse_video_generated----")
    if not os.path.isfile(path+name+'_naive_hyperlapse.mp4'):
        generate_hyperlapse_video(name, frames, path, "naive")
        print("----naive_hyperlapse_video_generated----")
    if not os.path.isfile(path+'optimal/'+name+'_0.jpg'):
        generate_frame_images(name, frames, path+'optimal/',"optimal")
        print("----optimal_frame_images_generated----")
    if not os.path.isfile(path+'naive/'+name+'_0.jpg'):
        generate_frame_images(name, frames, path+'naive/',"naive")
        print("----naive_frame_images_generated----")
    if not os.path.isfile('../npy/'+name+'_original_diff.npy'):
        generate_image_difference_data(name, frames)
        print("----image_difference_generated----")

def extract_sift_features(name, frames, num_frames):
    """
    Extract SIFT features from video frames and save them in numpy files with one file.
    f represents keypoints matrix and has shape [num_keypoints, 4], where the second index stores 
    Y-coord, X-coord, scale_factor, and orientation_factor respectively.
    d represents descriptors matrix and has shape [num_keypoints, 128].

    Args:
        name: name of the video chosen
        frames: video frames in numpy array, shape=[num_frames,height_frame,width_frame]
        num_frames: number of frames in total in the original frame sequence
    Returns nothing
    """
    f = np.zeros((num_frames, 100, 4))
    d = np.zeros((num_frames, 100, 128))
    for i in range(num_frames):
        _, (f_i, d_i) = utils.extract_sift_features(frames[i], scale=hp.down_scale, num_keypoints=100)
        f[i] = f_i
        d[i] = d_i
    np.save('../npy/'+name+"_f",f)
    np.save('../npy/'+name+"_d",d)

def select_optimal_frames(name, frames, T,w=hp.window_size,g=hp.gap_size):
    """
    Select optimal frame path from frame sequence and save the path in a numpy file.
    First, it initializes a static cost matrix. Then, it traverses through the static cost
    matrix and calculate a dynamic cost matrix. Lastly, it goes over the dynamic cost matrix
    and find a minimal cost path.

    Args:
        name: name of the video chosen
        frames: video frames in numpy array, shape=[num_frames,height_frame,width_frame]
        T: number of frames in total in the original frame sequence
        w: window size used in dynamic programming, i.e. the number of frames ahead that each
            frame compares to, default value set in hyperparamers.py
        g: gap size used when initializing cost matrix, i.e. the number of frames in which the
            initial and terminal frame is selected in, default value set in hyperparamers.py
    Returns nothing
    """
    # preparing keypoint and descriptor matrices
    f = np.float32(np.load('../npy/'+name+'_f.npy'))
    d = np.float32(np.load('../npy/'+name+'_d.npy'))
    cp = np.float32([frames[0].shape[0]*hp.down_scale/2, frames[0].shape[1]*hp.down_scale/2, 1])
    dynamic_cost = np.zeros((T, T))
    trace_back = np.zeros((T, T))
    # initializing dynamic cost matrix
    for i in range(T):
        for j in range(i+1, i+w):
            if (j < T):
                dynamic_cost[i,j] = compute_motion_cost(f,d,i,j,cp) + hp.gama_v * compute_velocity_cost(i,j)
    static_cost = np.copy(dynamic_cost)
    print("dynamic cost matrix initialized")
    # populating dynamic cost matrix
    for i in range(g, T):
        for j in range(i+1, i+w):
            if (j < T):
                optimal_k = 0
                optimal_cost = inf
                for k in range(1,w):
                    running_cost = dynamic_cost[i-k,i] + hp.gama_a * compute_acceleration_cost(i-k,i,j)
                    if running_cost < optimal_cost:
                        optimal_k = k
                        optimal_cost = running_cost
                dynamic_cost[i,j] = static_cost[i,j] + optimal_cost
                trace_back[i,j] = i - optimal_k          
    print("dynamic cost matrix populated")
    # locating terminal frame, hence the base case for dynamic programming
    start, dest = 0, 0
    min_cost = inf
    for i in range(T-g, T):
        for j in range(i+1, i+w):
            if (j < T):
                if (dynamic_cost[i,j] < min_cost):
                    min_cost = dynamic_cost[i,j]
                    start, dest = i, j
    # tracing back minimal cost path through dynamic programming
    path = [dest]
    trace_back = trace_back.astype(int)
    while (start > g):
        path = [start] + path
        back = trace_back[start,dest]
        dest = start
        start = back
    print("min cost path traced to end")
    np.save('../npy/'+name+"_p",path)

def compute_motion_cost(f,d,i,j,cp,p=hp.num_keyPoints):
    """
    Compute motion cost for between two given frames.

    Args:
        f: keypoint matrix of shape [num_frames, num_keypoints, 4], where the second index stores 
            Y-coord, X-coord, scale_factor, and orientation_factor respectively
        d: descriptor matrix of shape [num_frames, num_keypoints, 128]
        i: any frame index between 0 and T
        j: any frame index between i+1 and i+w (w=window_size)
        cp: coordinates of the center point of equally downscaled image frame in shape [Y,X] 
        p: number of keypoints taken into account, default value set in hyperparameters.py

    Returns:
        motion_cost: a float value of the motion cost between the given frames
    """
    motion_cost = 0
    # matching features and calculating homography
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = sorted(bf.match(d[i],d[j]), key = lambda x:x.distance) # matching and sorting
    matches = matches[:p] # filtering
    kpi = np.delete(np.float32([ f[i,m.queryIdx] for m in matches ]),3,1) # getting keypoint coords
    kpj = np.delete(np.float32([ f[j,m.trainIdx] for m in matches ]),3,1)
    kpi[:,2], kpj[:,2]  = np.ones(p), np.ones(p) # creating [Y,X,1] location vector fo findHomography
    homography,_ = cv2.findHomography(kpi, kpj, cv2.RANSAC, 3)
    # calculating components of motion cost
    alignment_cost = sum([norm(kpj[k].T - homography * kpi[k].T) for k in range(p)]) / p
    overlap_cost = norm(cp.T - homography * cp.T)
    motion_cost = overlap_cost
    if alignment_cost > hp.motion_threshold:
        motion_cost = hp.motion_max
    return motion_cost

def compute_velocity_cost(i, j, v=hp.speedup_rate):
    """
    Computes velocity cost between two given frames.

    Args:
        i: any frame index between 0 and T
        j: any frame index between i+1 and i+w (w=window_size)
        v: the target speedup rate we want to match in the output hyperlapse,
            default value set in hyperlapse.py

    Returns:
        a float value of the velocity cost between the given frames
    """
    return min([norm(j - i - v), hp.velocity_threshold])

def compute_acceleration_cost(h, i, j):
    """
    Computes accelaration cost between two given frames.

    Args:
        h: the number of frames skipped over by i
        i: any frame index between 0 and T
        j: any frame index between i+1 and i+w (w=window_size)

    Returns:
        - a float value of the acceleration cost between the given frames
    """
    return min([norm((j - i) - (i - h)),hp.acceleration_threshold])

def generate_hyperlapse_video(name, frames, path, option):
    """
    Generates and writes hyperlapse videos from the given frame path.
    If option is "naive", it will output the naive hyperlapse in which frame path is taken
    uniformly at random. If option is "optimal", it will output the optimal hyperlapse using
    the optimal path computed and stored beforehand.

    Args:
        name: name of the video chosen
        frames: video frames in numpy array, shape=[num_frames,height_frame,width_frame]
        path: the file path to which the output should be stored in 
        option: "naive" or "optimal", indicating which frame path to use

    Returns nothing
    """
    p = np.int32(np.load('../npy/'+name+'_p.npy'))
    num_frames = len(p)
    (img_h, img_w, img_c) = frames[0].shape
    hyperlapse = np.zeros((num_frames, img_h, img_w, img_c))
    if (option == "naive"):
        p = np.linspace(0,len(frames),num=num_frames,endpoint=False,dtype=int)
    for i in range(num_frames):
        print(p[i])
        hyperlapse[i] = frames[p[i]] 
    skvideo.io.vwrite(path+name+'_'+option+'_hyperlapse.mp4', hyperlapse)

def generate_frame_images(name, frames, path, option):
    """
    Generates and writes hyperlapse image sequence from the given frame path.
    If option is "naive", it will output the naive hyperlapse in which frame path is taken
    uniformly at random. If option is "optimal", it will output the optimal hyperlapse using
    the optimal path computed and stored beforehand.

    Args:
        name: name of the video chosen
        frames: video frames in numpy array, shape=[num_frames,height_frame,width_frame]
        path: the file path to which the output should be stored in 
        option: "naive" or "optimal", indicating which frame path to use

    Returns nothing
    """
    p = np.int32(np.load('../npy/'+name+'_p.npy'))
    num_frames = len(p)
    (img_h, img_w, img_c) = frames[0].shape
    hyperlapse = np.zeros((num_frames, img_h, img_w, img_c))
    if (option == "naive"):
        p = np.linspace(0,len(frames),num=num_frames,endpoint=False,dtype=int)
    for i in range(num_frames):
        skimage.io.imsave(path+name+'_'+str(i)+'.jpg',frames[p[i]])

def generate_image_difference_data(name, frames):
    """
    Computes and generates the difference between naive approach and optimal approach 
    with different distance metrics. This is for illustration purpose only and not needed
    for the pipeline itself.

    Args:
        name: name of the video chosen
        frames: video frames in numpy array, shape=[num_frames,height_frame,width_frame]

    Returns nothing
    """
    optimal_path = np.int32(np.load('../npy/'+name+'_p.npy'))
    naive_path = np.linspace(0,len(frames),num=len(optimal_path),endpoint=False,dtype=int)

    original_num_frames = len(frames)
    hyperlapse_num_frames = len(optimal_path)

    optimal_diff = np.float32(np.zeros((hyperlapse_num_frames)))
    naive_diff = np.float32(np.zeros((hyperlapse_num_frames)))
    
    optimal_frames = np.zeros((hyperlapse_num_frames, 1080, 1920, 3))
    naive_frames = np.zeros((hyperlapse_num_frames, 1080, 1920, 3))
    for i in range(hyperlapse_num_frames):
        optimal_frames[i] = frames[optimal_path[i]]
        naive_frames[i] = frames[naive_path[i]]
    
    optimal_keypoint = np.float32(np.zeros((hyperlapse_num_frames, 100, 4)))
    optimal_descriptor = np.float32(np.zeros((hyperlapse_num_frames, 100, 128)))
    naive_keypoint = np.float32(np.zeros((hyperlapse_num_frames, 100, 4)))
    naive_descriptor = np.float32(np.zeros((hyperlapse_num_frames, 100, 128)))

    for i in range(hyperlapse_num_frames):
        _, (nf_i, nd_i) = utils.extract_sift_features(naive_frames[i], scale=hp.down_scale, num_keypoints=100)
        _, (of_i, od_i) = utils.extract_sift_features(optimal_frames[i], scale=hp.down_scale, num_keypoints=100)
        naive_keypoint[i] = nf_i
        naive_descriptor[i] = nd_i
        optimal_keypoint[i] = of_i
        optimal_descriptor[i] = od_i
    
    cp = np.float32([frames[0].shape[0]*hp.down_scale/2, frames[0].shape[1]*hp.down_scale/2, 1])
    naive_diff = compute_image_difference(naive_keypoint,naive_descriptor,cp,hyperlapse_num_frames)
    optimal_diff = compute_image_difference(optimal_keypoint,optimal_descriptor,cp,hyperlapse_num_frames)

    np.save('../npy/'+name+"_naive_diff",naive_diff)
    np.save('../npy/'+name+"_naive_path",naive_path)
    np.save('../npy/'+name+"_optimal_diff",optimal_diff)
    np.save('../npy/'+name+"_optimal_path",optimal_path)

def compute_image_difference(keypoint, descriptor, cp, num_frames, num_keypoints=hp.num_keyPoints):
    """
    Computes and the image difference between consecutive frames with different distance metrics. 
    This is for illustration purpose only and not neededfor the pipeline itself.

    Args:
        keypoint: keypoint matrix of shape [num_frames, num_keypoints, 4], where the second index stores 
            Y-coord, X-coord, scale_factor, and orientation_factor respectively
        descriptor: descriptor matrix of shape [num_frames, num_keypoints, 128]
        cp: coordinates of the center point of equally downscaled image frame in shape [Y,X] 
        num_frames: number of frames in the frame sequence
        num_keypoints: number of keypoints taken into account, default value set in hyperparameters.py

    Returns:
        difference vector computed using given keypoints and descriptors
    """
    
    difference = np.zeros((num_frames))
    for i in range(num_frames-1):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = sorted(bf.match(descriptor[i],descriptor[i+1]), key = lambda x:x.distance) # matching and sorting
        matches = matches[:num_keypoints] # filtering
        kpi = np.delete(np.float32([ keypoint[i,m.queryIdx] for m in matches ]),3,1) # getting keypoint coords
        kpj = np.delete(np.float32([ keypoint[i+1,m.trainIdx] for m in matches ]),3,1)
        kpi[:,2], kpj[:,2]  = np.ones(num_keypoints), np.ones(num_keypoints)
        homography,_ = cv2.findHomography(kpi, kpj, cv2.RANSAC, 3)
        difference[i] = norm(cp.T - homography * cp.T) # center distance
        # difference[i] = sum([norm(kpj[k].T - homography * kpi[k].T) for k in range(num_keypoints)]) / num_keypoints # projected_feature_distance
        # difference[i] = sum([norm(kpj[k].T - kpi[k].T) for k in range(num_keypoints)]) / num_keypoints # feature_distance
    return difference


if __name__ == '__main__':
    main()
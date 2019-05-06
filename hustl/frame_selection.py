import utils
import hyperparameters as hp
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
    if not os.path.isfile(path+name+'_0.jpg'):
        generate_optimal_frame_images(name, frames, path)
        print("----optimal_frame_images_generated----")

def extract_sift_features(name, frames, num_frames):
    f = np.zeros((num_frames, 100, 4))
    d = np.zeros((num_frames, 100, 128))
    for i in range(num_frames):
        _, f_i, d_i = utils.extract_sift_features(frames[i], scale=hp.down_scale, num_keypoints=100)
        f[i] = f_i
        d[i] = d_i
    np.save('../npy/'+name+"_f",f)
    np.save('../npy/'+name+"_d",d)

def select_optimal_frames(name, frames, T,w=hp.window_size,g=hp.gap_size):
    # preparing keypoint and descriptor matrices
    f = np.float32(np.load('../npy/'+name+'_f.npy'))
    d = np.float32(np.load('../npy/'+name+'_d.npy'))
    cp = img_ct_coord = np.float32([frames[0].shape[0]*hp.down_scale/2, frames[0].shape[1]*hp.down_scale/2, 1])
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
    # locating last frame
    start, dest = 0, 0
    min_cost = inf
    for i in range(T-g, T):
        for j in range(i+1, i+w):
            if (j < T):
                if (dynamic_cost[i,j] < min_cost):
                    min_cost = dynamic_cost[i,j]
                    start, dest = i, j
    # tracing back min cost path
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
    motion_cost = 0
    # matching features and calculating homography
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = sorted(bf.match(d[i],d[j]), key = lambda x:x.distance) # matching and sorting
    matches = matches[:p] # filtering
    kpi = np.delete(np.float32([ f[i,m.queryIdx] for m in matches ]),3,1) # getting keypoint coords
    kpj = np.delete(np.float32([ f[j,m.trainIdx] for m in matches ]),3,1)
    kpi[:,2], kpj[:,2]  = np.ones(p), np.ones(p)
    homography,_ = cv2.findHomography(kpi, kpj, cv2.RANSAC, 3)
    # calculating components of motion cost
    alignment_cost = sum([norm(kpj[k].T - homography * kpi[k].T) for k in range(p)]) / p
    overlap_cost = norm(cp.T - homography * cp.T)
    motion_cost = overlap_cost
    if alignment_cost > hp.motion_threshold:
        motion_cost = hp.motion_max
    return motion_cost

def compute_velocity_cost(i, j, v=hp.speedup_rate):
    return min([norm(j - i - v), hp.velocity_threshold])

def compute_acceleration_cost(h, i, j):
    return min([norm((j - i) - (i - h)),hp.acceleration_threshold])

def generate_hyperlapse_video(name, frames, path, option):
    p = np.int32(np.load('../npy/'+name+'_p.npy'))
    num_frames = len(p)
    (img_h, img_w, img_c) = frames[0].shape
    hyperlapse = np.zeros((num_frames, img_h, img_w, img_c))
    if (option == "naive"):
        p = np.linspace(0,len(frames),num=num_frames,endpoint=False,dtype=int)
    for i in range(num_frames):
        hyperlapse[i] = frames[p[i]]
    skvideo.io.vwrite(path+name+'_'+option+'_hyperlapse.mp4', hyperlapse)

def generate_optimal_frame_images(name, frames, path):
    p = np.int32(np.load('../npy/'+name+'_p.npy'))
    num_frames = len(p)
    (img_h, img_w, img_c) = frames[0].shape
    hyperlapse = np.zeros((num_frames, img_h, img_w, img_c))
    for i in range(num_frames):
        skimage.io.imsave(path+name+'_'+str(i)+'.jpg',frames[p[i]])


if __name__ == '__main__':
    main()
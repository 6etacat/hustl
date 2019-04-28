import numpy as np
import cv2
import cyvlfeat as vlfeat
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte
from skimage.transform import resize, rescale
from scipy import ndimage
import os
import subprocess
import scipy
import visualize
import re

def estimation(files, names, scale):
    print("starting estimation")

    O = np.load('../npy/observation.npy')
    O, W, _, _ = initialization(O)

    lbd = 1/np.sqrt(np.amin([O[0].shape[0], O[0].shape[1]]))
    rho = 1.05 ## do not change

    num_img = O[0].shape[1]
    num_pts = O[0].shape[0]

    albedo, const, gamma = [], [], []
    O_low = []

    for ch in range(0,3):
        a_init = init_albedo(np.ones(num_img), np.zeros(num_img), O[ch], W)
        g_init = np.ones(num_img)
        c_init = np.zeros(num_img)

        A = np.array([a_init, np.ones(num_pts)])
        GC = np.array([g_init, np.multiply(c_init, g_init)])

        l1_rpca_mask_alm_fast(O[ch], W, A, 2, lbd, A, GC.T, 1, rho, scale)

        # GC_est = GC_est.T
        # O_low[ch] = O_l
        # albedo[ch] = np.exp(A_est[:, 1])
        # gamma[ch] = GC_test[1, :]
        # const[ch] = np.exp(GC_test[2, :] / GC_test[1, :])


def initialization(O):
    print("initializing")

    #used to initialize albedo, gamma, constants, and indicator mat
    num_img = O[0].shape[1]
    num_pts = O[0].shape[0]

    gamma = []
    cons = []
    W = np.zeros((num_pts, num_img))
    v_id = []

    for ch in range(0,3):
        gamma.append(np.ones(num_img))
        cons.append(np.zeros(num_img))
        temp = np.where(O[ch] > 2/255)[0]
        v_id.append(temp)

    gamma = np.array(gamma)
    cons = np.array(cons)

    v_id_intersect = np.intersect1d(np.intersect1d(v_id[0], v_id[1]), v_id[2])
    W[v_id_intersect] = 1

    #### visualizing imgs
    # implement later

    #### log scaling img
    for ch in range(0,3):
        sm_o = O[ch]
        O[ch, v_id_intersect] = np.log(sm_o[v_id_intersect])

    print("initialization completed")
    return O, W, gamma, cons

def init_albedo(pg, pc, O, W):
    print("initializaing albedo")

    num_pts = O.shape[0]
    b = np.zeros(num_pts)

    for pt_id in range(0, num_pts):
        v_id_init = np.where(W[pt_id, :] == 1)[0]
        n_vid_init = v_id_init.size

        if n_vid_init == 0:
            continue

        ades = np.zeros(n_vid_init)
        for iter in range(0, n_vid_init):
            img_id = v_id_init[iter]
            gg = pg[img_id]
            cc = pc[img_id]
            oo = O[pt_id, img_id]
            ades[iter] = oo / gg - cc

        ades_mid = np.median(ades)
        b[pt_id] = ades_mid

    albedo = b
    print("completed initializaing albedo")
    return albedo

def l1_rpca_mask_alm_fast(M, W, Ureg, r, lbd1, U, B, maxIterIN, rho, scale):
    """ This code is based on the MATLAB implementation by Ricardo Carbal
        of the paper Unifying Nuclear Norm and Bilinear Factorization Approaches
        for Low-rank Matrix Decomposition
    """
    print("starting l1_rpca_mask_alm_fast()")

    ## can add GPU option. not coded yet

    m, n = M.shape[0], M.shape[1]
    maxIterOut = 5000
    max_mu = 1e20
    mu = 1e-3
    M_norm = np.linalg.norm(M, 'fro')
    tol = 1e-9 * M_norm

    cW = np.ones(W.size) - W.ravel()
    is_display_progress = True

    #### initializing optimization var as zeros
    E = np.random.normal(size=(m,n))
    Y = np.zeros((m,n)) #lagrange multiplier
    Y = M
    _,norm_two,_ = np.linalg.svd(Y)
    norm_two = norm_two[0]

    mu = 1.25 / norm_two
    norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lbd1
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    exit()

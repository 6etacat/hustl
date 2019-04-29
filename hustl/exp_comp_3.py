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
from numpy.linalg import norm, inv

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

        A = np.column_stack([a_init, np.ones((num_pts ,1))])
        GC = np.vstack([g_init, np.multiply(c_init, g_init)]) # their GC is vertical

        O_l, A_est, GC_est, obj = l1_rpca_mask_alm_fast(O[ch], W, A, 2, lbd, A, GC.T, 1, rho, scale)

        GC_est = GC_est.T
        O_low.append(O_l)
        albedo.append(np.exp(A_est[:, 0]))
        gamma.append(GC_est[0, :])
        const.append(np.exp(GC_est[1, :] / GC_est[0, :]))

    O_low = np.array(O_low)
    albedo = np.array(albedo)
    const = np.array(const)
    gamma = np.array(gamma)

    print("estimation completed, saving files")
    np.savez('../npy/estimation', O_low=O_low, albedo=albedo, const=const, gamma=gamma)

def apply(files, names):

    print("begin applying everything")

    estimations = np.load('../npy/estimation.npz')
    O_low = estimations['O_low']
    albedo = estimations['albedo']
    const = estimations['const']
    gamma = estimations['gamma']

    num_frames = len(files)
    if num_frames == 2:
        is_style_transfer = True
        transfer_id = 1
    else:
        is_style_transfer = False
        transfer_id = 1

    for i in range(0, num_frames):
        img = io.imread(files[i])
        img = rescale(img, 6/23, anti_aliasing=True, multichannel=True, mode='reflect')

        img = img_as_float(img)
        for ch in range(0,3):
            cc = const[ch, i]
            gg = gamma[ch, i]
            ii = img[:, :, ch]

            ### prevent abrupt change
            if (gg < 0.5) or (gg > 3):
                gg = 1

            if (cc < 0.3) or (cc > 3):
                cc = 1

            ### style transfer
            if is_style_transfer:
                ii = ii ** (1/gg) / cc
                cc2 = const[ch, transfer_id]
                gg2 = gamma[ch, transfer_id]
                ii = (ii * cc2) ** gg2
                ii = np.clip(ii, -1, 1)
                # ii[ii > 1] = 1
                # ii[ii < 0] = 0
                img[:, :, ch] = ii

            else:
                img[:, :, ch] = (ii ** (1/gg)) / cc
                img = np.clip(img, -1, 1)

        img = img_as_ubyte(img)
        plt.imshow(img)
        plt.show()



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
        O[ch, v_id_intersect] = np.log(sm_o[v_id_intersect] + 0.000000001)

    print("initialization completed")
    return O, W, gamma, cons

def init_albedo(pg, pc, O, W):
    print("initializaing albedo")

    num_pts = O.shape[0]
    b = np.zeros((num_pts,1))

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

def l1_rpca_mask_alm_fast(M, W, Ureg, r, lbd1, U, V, maxIterIN, rho, scale):
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
    M_norm = norm(M, 'fro')
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
    norm_inf = norm(Y.ravel(), np.inf) / lbd1
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    ## caching
    lr1 = lbd1 * np.eye(r, dtype=int)
    lbd2 = lbd1 * scale
    lr2 = lbd2 * np.eye(r, dtype=int)

    ### start main outer loop
    print("starting main outer loop")
    iter_OUT = 0
    while iter_OUT < maxIterOut:
        iter_OUT = iter_OUT+1

        itr_IN = 0
        obj_pre = 1e20

        ### start inner loop
        while itr_IN < maxIterIN:

            # update U
            tmp = mu * E + Y
            U = (tmp @ V + lbd2 * Ureg) @ inv((lr1 + mu*(V.T @ V) + lr2))
            U[:, 1] = 1

            # update V
            V = tmp.T @ U @ inv((lr1 + mu*(U.T @ U)))

            # update E
            UV = U @ V.T
            temp1 = UV - Y/mu

            #l1
            temp = M - temp1
            El1 = np.clip(temp - 1/mu, 0, None) + np.clip(temp + 1/mu, None, 0)
            El1 = (M-El1)

            E = El1 * W + temp1 * cW.reshape(temp1.shape[0], temp1.shape[1])

            # evaluate current objective
            temp1 = np.sum(W * np.abs(M-E))
            temp2 = norm(U, 'fro') ** 2
            temp3 = norm(V, 'fro') ** 2
            temp4 = np.sum(Y * (E-UV))
            temp5 = norm(E-UV, 'fro') ** 2
            temp6 = norm(U-Ureg, 'fro') ** 2
            obj_cur = temp1 + lbd1/2*temp2 + temp3 + temp4 + mu/2*temp5 + lbd2/2*temp6

            # check convergence of inner loop
            if np.abs(obj_cur - obj_pre) <= 1e-10 * np.abs(obj_pre):
                break
            else:
                obj_pre = obj_cur
                itr_IN = itr_IN + 1

            leq = E-UV
            stopC = norm(leq, 'fro')
            if stopC < tol:
                break
            else:
                # update lagrange multiplier
                Y = Y + mu * leq
                # update penalty parameter
                mu = min(max_mu, mu * rho)

            # denormalization
            U_est = U
            V_est = V

            M_est = U_est @ V_est
            obj = np.sum(W * np.abs(M-E)) + lbd1/2*(norm(U, 'fro')) + norm(V, 'fro')

    print("finished")
    return M_est, U_est, V_est, obj

import cv2 as cv
import numpy as np
import tifffile

import os, sys

from preprocessing import get_views


def roll_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [+s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def generate_objpoints(num_projs, dims=(8,6), roll_angle=0.0, tilt_angle=0.0):
    '''
    transz_step = +5.0  # mm
    sdd = 644  # mm
    sod_initial = 434  # mm
    sod = sod_initial + transz_step * np.arange(num_projs)
    M = sdd/sod
    rot_step = np.deg2rad(+3.0)
    yaw_initial = np.deg2rad(-15.0)
    yaw_mtxs = np.stack([_yaw_matrix(yaw_initial + n * rot_step) 
                         for n in range(num_projs)])
    '''
    spacing = 14.22  # mm
    sod = 589  # mm
    sdd = 679  # mm
    M = sdd/sod

    total = dims[0]*dims[1]
    objp = np.zeros((num_projs, total, 3), np.float32)
    objp[:,:,:2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)
    objp *= spacing * M
    centroid = np.mean(objp[:,:,:2], axis=1)
    objp[:,:,:2] -= centroid[:,np.newaxis,:]
    objp[:,:,1] *= -1

    def _yaw_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [ c,  0.0,  -s],
            [0.0, 1.0, 0.0],
            [+s,  0.0,  c]
        ])

    def _tilt_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, +s, c]
        ])

    roll_mtx = roll_matrix(roll_angle)
    return objp.squeeze() @ roll_mtx.T


def get_imgpoints(path, dims_list=[(10,8)]):
    _, views = get_views(path)
    blackwhite = np.where(views < 2000, 255, 0).astype(np.uint8)

    imgpoints = []
    for i, bw in enumerate(blackwhite):
        bw = cv.medianBlur(bw, 5)
        if os.environ.get("VIZ") == "2":
            import matplotlib.pyplot as plt
            plt.imshow(bw, cmap='gray')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.show()

        found = False
        centers_refined = None
        dims_used = None
        for dims in dims_list:
            ret, centers = cv.findCirclesGrid(
                bw, dims,
                flags=cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING
            )
            if ret:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                centers_refined = cv.cornerSubPix(bw, centers, (11, 11), (-1, -1), criteria)
                dims_used = dims
                found = True
                imgpoints.append(centers_refined)
                break
        if not found:
            print(f'No grid found in view {i}')
    if os.environ.get("VIZ") == "1":
        print(imgpoints)
    return dims, np.array(imgpoints).squeeze().squeeze()


def estimate_roll_2d(srcpoints, imgpoints):
    """
    Estimate roll angle (in radians) that best aligns srcpoints -> imgpoints
    via least squares rotation fit.
    """
    src_c = srcpoints
    dst_c = imgpoints - imgpoints.mean(axis=0)

    # Compute 2×2 covariance
    cov = src_c.T @ dst_c

    # SVD for optimal rotation
    U, _, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Roll angle around Z (in-plane)
    roll = np.arctan2(R[1, 0], R[0, 0])
    return roll, R



if __name__ == "__main__":
    ''' work in progress
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (2200,2750), None, None)
    '''
    if len(sys.argv) < 2:
        print("Usage: python camera_matrix.py <dir_with_tiffs>")
        sys.exit(1)

    dir_path = sys.argv[1]

    dims, imgpoints = get_imgpoints(dir_path)
    srcpoints = generate_objpoints(1, dims)[:,:2]

    roll, R = estimate_roll_2d(srcpoints, imgpoints)
    my_roll = 180 - np.rad2deg(roll)  # deg

    my_R = roll_matrix(-np.deg2rad(my_roll))
    H, mask = cv.findHomography(srcpoints @ my_R[:2,:2].T, imgpoints, cv.RANSAC)
    np.save("../data/H_mtx_negative", H)

    my_R = roll_matrix(np.deg2rad(my_roll))
    H, mask = cv.findHomography(srcpoints @ my_R[:2,:2].T, imgpoints, cv.RANSAC)
    np.save("../data/H_mtx_positive", H)


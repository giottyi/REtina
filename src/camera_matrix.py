import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tifffile

import sys

from preprocessing import get_views


def _generate_objpoints(num_projs, dims=(8,6)):
    spacing = 14.22  # mm
    transz_step = +5.0  # mm
    rot_step = np.deg2rad(+3.0)
    sdd = 644  # mm
    sod = 434 + transz_step * np.arange(num_projs)
    M = sdd/sod

    total = dims[0]*dims[1]
    objp = np.zeros((num_projs, total, 3), np.float32)
    print(objp.shape)
    objp[:,:,:2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)
    objp[:,:,:2] *= spacing * M  # mm
    centroid = np.mean(objp[:,:,:2], axis=1)
    objp[:,:,:2] -= centroid
    objp[:,:,1] *= -1
    return objp
    #return np.broadcast_to(objp, (num_projs,total,3))


def _get_imgpoints(path, dims_list=[(8,6)]):
    _, views = get_views(path)
    views[:,:200,:] = 0
    views[:,:,:150] = 0
    blackwhite = np.where(views < 2000, 255, 0).astype(np.uint8)

    imgpoints = []
    for i, bw in enumerate(blackwhite):
        bw = cv.medianBlur(bw, 5)

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
    return imgpoints


''' def main():
    H, mask = cv2.findHomography(objp_rotated[:,:-1], corners_subpix, cv2.RANSAC)
    sdd = 644  # mm
    transz_jog = 5  # mm
    sod =  434 + transz_jog * np.arange(num_projs)
    def _roll_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [+s, c, 0],
            [0, 0,  1]]) '''
    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python camera_matrix.py <dir_with_tiffs>")
        sys.exit(1)

    dir_path = sys.argv[1]
    imgpoints = _get_imgpoints(dir_path)
    objpoints = _generate_objpoints(len(imgpoints))
    #ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (2200,2750), None, None)


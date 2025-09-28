import os
os.environ["OMP_NUM_THREADS"] = '16'

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import glob, sys#, time, datetime as dt

from preprocessing import get_views, get_flat


def ICS_normalize(src, flat):
    corrected = (src / flat).astype(np.float32)
    filtered = cv.bilateralFilter(corrected, d=9, sigmaColor=75, sigmaSpace=75)
    normalized = (filtered - filtered.min()) \
            / (filtered.max() - filtered.min()) * 255
    return normalized.astype(np.uint8)


def detect_centers(src, flat):
    """ 
    cv.HoughCircles parameters are optimized for flat-corrected projections, 
    if needed add:
    src = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)
    src = cv.equalizeHist(src)
    """
    src = ICS_normalize(src, flat)
    circles = cv.HoughCircles(src, cv.HOUGH_GRADIENT, dp=1, \
            minDist=src.shape[0]/16, param1=51, param2=25, \
            minRadius=20, maxRadius=120)
    if circles is None:
        print("WARNING: NO CENTERS DETECTED")
        return None
    
    circles = np.uint16(np.around(circles))
    centers = np.array([[i[0], i[1]] for i in circles[0,:]])
    return centers


def get_ICS_coords(projs_directory, flats_directory):
    """
    returns homogeneous coordinates of the beads on the retinal plane,
    following them between projections by sorting y coordinates
    """
    projs_angles, projs_stack = get_views(projs_directory)
    flat = get_flat(flats_directory)
    all_centers = {}
    for i, proj in enumerate(projs_stack):
        centers = detect_centers(proj, flat)
        if os.environ.get('VIZ') == '2' and i % 29 == 0:
            plot_centers(ICS_normalize(proj,flat), centers)
        if centers is not None:
            all_centers[projs_angles[i]] = centers

    num_centers = max([len(centers) for centers in all_centers.values()])
    homo_centers = np.full((num_centers, 3, len(projs_stack)), np.nan)
    homo_fill = np.ones_like(homo_centers[:,0,0])
    for proj_idx, proj_angle in enumerate(all_centers.keys()):
        centers = all_centers[proj_angle]
        centers = centers[np.argsort(-centers[:,-1])]
        homo_centers[:,0:2,proj_idx] = centers
        homo_centers[:,-1,proj_idx] = homo_fill
    return homo_centers


def plot_centers(src, centers=None):
    if centers is not None:
        for center in centers:
            cv.circle(src, center, 9, (255, 0, 0), 3)

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(src)
    ax.axis("off")
    ax.invert_yaxis()
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python beads_images.py <projs_dir> <flats_dir>")
        sys.exit(1)
    projs_dir = sys.argv[1]
    flats_dir = sys.argv[2]

    print("Analysing calibration data ...")
    projs_array = get_ICS_coords(projs_dir, flats_dir)
    os.makedirs('../data', exist_ok=True)
    np.save("../data/phantom_ICS.npy", projs_array)
    print(f"Saved normalized phantom projections as \'phantom_ICS.npy\' in ../data/")
    if os.environ.get('VIZ') == '1':
        print(projs_array)


    
if  __name__ == "__main__":
    main()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tifffile

import sys


path = "../data/grid_corrected.tif"

sod = 589  # mm
sdd = 679  # mm
M = sdd/sod
circle_spacing = 14.22 * M  # mm (center-to-center)

#hdul = fits.open(sys.argv[1])
#hdul.info()
#gray = hdul[0].data
stack = tifffile.imread(path)
gray = np.median(stack, axis=0)
if gray is None:
    raise ValueError(f"Failed to read image: {img_path}")

bw = np.where(gray < 2100, 255, 0).astype(np.uint8)
bw = cv2.medianBlur(bw, 5)

plt.imshow(bw, cmap='gray')
plt.axis('off')
#plt.savefig('grid.eps', bbox_inches='tight')
#plt.show()

candidate_dims = []
for cols in range(12, 8, -1):
    for rows in range(10, 6, -1):
        candidate_dims.append((cols, rows))

found = False
corners_subpix = None
dims_used = None
for dims in candidate_dims:
    print(f"Trying grid size: {dims}")
    ret, centers = cv2.findCirclesGrid(
        bw, dims,
        flags=cv2.CALIB_CB_SYMMETRIC_GRID) # + cv2.CALIB_CB_CLUSTERING
    if ret:
        print(f"✔ Found grid with size {dims}")
        dims_used = dims
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(bw, centers, (11,11), (-1,-1), criteria)
        found = True
        break

if not found:
    print("❌ No grid detected with any of the given sizes.")
    exit()

objp = np.zeros((dims_used[0]*dims_used[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dims_used[0], 0:dims_used[1]].T.reshape(-1, 2)
objp[:, :2] *= circle_spacing
centroid = np.mean(objp[:, :2], axis=0)
objp[:, :2] -= centroid
objp_2d = objp[:, :2]

H, mask = cv2.findHomography(objp_2d, corners_subpix, cv2.RANSAC)
print("Homography matrix:\n", H)


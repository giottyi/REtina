import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "../data/exp_022_crop.jpg"   # change to your .tif if needed
square_size = 8.24  # mm per square

gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
if gray is None:
    raise ValueError(f"Failed to read image: {img_path}")

gray = cv2.equalizeHist(gray)
#gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
#_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(gray, cmap='gray')
plt.show()

candidate_dims = []

for cols in range(14, 9, -1):
    for rows in range(12, 6, -1):
        if rows < cols:            # only valid if rows < cols
            candidate_dims.append((cols, rows))


found = False
corners_subpix = None
dims_used = None

for dims in candidate_dims:
    print(f"Trying checkerboard size: {dims}")
    ret, corners = cv2.findChessboardCorners(
        gray, dims,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    )
    if ret:
        print(f"✔ Found checkerboard with size {dims}")
        dims_used = dims
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        print(corners_subpix)
        found = True
        break

if not found:
    print("❌ No checkerboard detected with any of the given sizes.")
    exit()

# --- build real world coordinates
objp = np.zeros((dims_used[0]*dims_used[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dims_used[0], 0:dims_used[1]].T.reshape(-1, 2)
objp *= square_size
objp_2d = objp[:, :2]
print(objp_2d)

# --- Homography
H, mask = cv2.findHomography(objp_2d, corners_subpix, cv2.RANSAC)
print("Homography matrix:\n", H)

# --- visualize
vis = cv2.drawChessboardCorners(gray, dims_used, corners_subpix, True)
#cv2.imshow("Detected Checkerboard", vis)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


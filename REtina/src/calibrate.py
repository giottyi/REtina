
import os
os.environ["OMP_NUM_THREADS"] = '16'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

import os, sys

from preprocess import get_views, get_flat


def normalise_ics(src):
    corrected = np.log(src / src.max() + 1e-6).astype(np.float32)
    filtered = cv.GaussianBlur(corrected, ksize=(7,7), sigmaX=1.5, sigmaY=1.5)
    normalised = (filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255
    return normalised.astype(np.uint8)


def detect_centers(src):
    '''
    cv.HoughCircles parameters are optimized for flat-corrected projections, 
    if needed add:
    src = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)
    src = cv.equalizeHist(src)
    '''
    src = normalise_ics(src)
    circles = cv.HoughCircles(src, cv.HOUGH_GRADIENT_ALT, dp=1.5, \
            minDist=src.shape[0]/16, param1=300, param2=0.9, \
            minRadius=20, maxRadius=120)
    if circles is None:
        print("WARNING: NO CENTERS DETECTED")
        return None
    
    circles = np.uint16(np.around(circles))
    centers = np.array([[i[0], i[1]] for i in circles[0,:]])
    return centers


def plot_centers(src, centers=None):
    if centers is not None:
        for center in centers:
            cv.circle(src, center, 9, (255, 0, 0), 3)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    im = ax.imshow(src, cmap='RdBu')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def get_ics_coords(projs_directory, need_plots=False):
    """
    returns homogeneous coordinates of the beads on the retinal plane,
    following them between projections by sorting y coordinates
    """
    _, projs_stack = get_views(projs_directory, views_type='projections')
    
    all_centers = []
    for i, proj in enumerate(projs_stack):
        centers = detect_centers(proj)
        if need_plots is True and i == 0:
            plot_centers(normalise_ics(proj), centers)
        all_centers.append(centers)

    num_centers = max([len(centers) for centers in all_centers])
    homo_centers = np.full((num_centers, 3, len(projs_stack)), np.nan)
    homo_fill = np.ones_like(homo_centers[:,0,0])
    for proj_idx, centers in enumerate(all_centers):
        centers = centers[np.argsort(centers[:,-1])]
        homo_centers[:,0:2,proj_idx] = centers
        homo_centers[:,-1,proj_idx] = homo_fill
    return homo_centers


def HPRt(params, sdd, H, num_projs):
    '''
    creates the OCS -> WCS -> DCS projection matrices for all views,
    linear model for the post-detector projective geometry, supposing
    known homography matrix. SDD must be know since minimazion only predicts
    magnification and offsets
    '''
    trans_x, trans_y, trans_z, tilt, roll, initial = params
    
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  np.cos(tilt), +np.sin(tilt)],
        [0.0, -np.sin(tilt),  np.cos(tilt)]
    ])

    R_z = np.array([
        [ np.cos(roll), +np.sin(roll), 0.0],
        [-np.sin(roll),  np.cos(roll), 0.0],
        [0.0, 0.0, 1.0]
    ])

    rot_step = np.deg2rad(360) / (num_projs - 1)
    R_y_fn = lambda n: np.array([ 
        [ np.cos(initial-n*rot_step), 0.0, +np.sin(initial-n*rot_step)],
        [0.0, 1.0, 0.0],
        [-np.sin(initial-n*rot_step), 0.0,  np.cos(initial-n*rot_step)]
    ])
    R_y_projs = np.stack([R_y_fn(n) for n in range(num_projs)])

    R = R_x @ R_z @ R_y_projs
    t = np.array([[trans_x], [trans_y], [trans_z]])
    t = np.tile(t[np.newaxis,:,:], (num_projs,1,1))
    Rt = np.concatenate((R,t), axis=2)
    homo_row = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (num_projs, 1, 1))
    Rt = np.concatenate((Rt, homo_row), axis=1)

    P = np.array([ 
        [sdd, 0.0, 0.0, 0.0],
        [0.0, sdd, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    return H @ P @ Rt


def homo_normalisation(res):
    eps = np.finfo(np.float16).tiny
    res[:,0,:] /= res[:,2,:] + eps
    res[:,1,:] /= res[:,2,:] + eps
    res[:,2,:] = 1.0
    return res


def normalise_params(params, param_scales, param_offsets):
    return (np.array(params) - param_offsets) / param_scales


def denormalise_params(norm_params, param_scales, param_offsets):
    return norm_params * param_scales + param_offsets


def normalise_bounds(bounds, param_scales, param_offsets):
    norm_bounds = []
    for i, (lower, upper) in enumerate(bounds):
        norm_lower = (lower - param_offsets[i]) / param_scales[i]
        norm_upper = (upper - param_offsets[i]) / param_scales[i]
        norm_bounds.append((norm_lower, norm_upper))
    return norm_bounds


def norm_objective_function(norm_params, ocs, ics, sdd, H, param_scales, param_offsets):
    '''
    normalized for better convergence and stability as in Hartley & Zisserman
    '''
    num_beads, _, num_projs = ics.shape
    params = denormalise_params(norm_params, param_scales, param_offsets)
    LPRt_matrix = HPRt(params, sdd, H, num_projs)
    cam = np.matmul(LPRt_matrix, ocs.transpose(2,1,0))
    cam = homo_normalisation(cam.transpose(2,1,0))
    return np.linalg.norm(cam - ics) / num_projs / num_beads


def adjust_ocs(ocs):
    '''
    rotates coordinates from OCS to WCS for consistency and translates
    into baricenter for symmetry
    '''
    old_y = ocs[:,1].copy()
    old_z = ocs[:,2].copy()
    ocs[:,0] *= -1
    ocs[:,2] = old_y
    ocs[:,1] = old_z
    centroid = np.mean(ocs, axis=0)
    ocs[:,:-1] -= centroid[:-1]
    return ocs


def scatter_plot(res):
    x = res[:,0,:]
    y = res[:,1,:]
    z = res[:,2,:]
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y)
    ax.plot(x.flatten(), z.flatten(), y.flatten(), color='gray', linewidth=1)
    ax.set_xlabel('u')
    ax.set_xlim((0,2750))
    ax.set_ylabel('Z')
    ax.set_zlabel('v')
    ax.set_zlim((0,2200))
    ax.invert_zaxis()
    plt.show()


def calibrate(projs_dir, sdd, need_plots=False):
    print('Sarting calibration ...')
    ocs = adjust_ocs(np.load("../calibration_data/phantom_ocs.npy"))[:-1]
    ics = get_ics_coords(projs_dir, need_plots)[:-1]
    H = np.load('../calibration_data/H_mtx.npy')

    num_projs = ics.shape[-1]

    if need_plots is True:
        scatter_plot(ics)

    initial_params = [
        0.0, 0.0,           # trans_x, trans_y
        500.0,              # trans_z, [mm]
        np.deg2rad(0.0),    # tilt
        np.deg2rad(0.0),    # roll
        np.deg2rad(0.0)     # initial rotation
    ]

    eps = np.finfo(np.float16).tiny
    bounds = [
        (-100.0, 100.0),            # trans_x, mm
        (-100.0, 100.0),            # trans_y, mm
        (eps, sdd),                 # trans_z, mm
        (-np.pi/4, np.pi/4),        # tilt, radians
        (-np.pi/4, np.pi/4),        # roll, radians
        (-2.0*np.pi, 2.0*np.pi)     # rot, radians
    ]

    param_scales = np.array([
        10.0,      # trans_x: mm
        10.0,      # trans_y: mm
        100.0,     # trans_z: mm
        np.pi/8,    # tilt: ±π/4 -> ±1
        np.pi/8,    # roll: ±π/4 -> ±1
        np.pi       # initial: ±π -> ±1
    ])

    param_offsets = np.array([
        0.0,        # trans_x: centered at 0
        0.0,        # trans_y: centered at 0
        500.0,     # trans_z: centered at 1000 mm
        0.0,        # tilt: centered at 0
        0.0,        # roll: centered at 0
        0.0         # initial: centered at 0
    ])

    if need_plots is True:
        LPRt_matrix = HPRt(initial_params, sdd, H, num_projs)
        cam = np.matmul(LPRt_matrix, ocs.transpose(2,1,0))
        cam = homo_normalisation(cam.transpose(2,1,0))
        scatter_plot(cam)
    
    norm_initial = normalise_params(initial_params, param_scales, param_offsets)
    norm_bounds = normalise_bounds(bounds, param_scales, param_offsets)
    norm_calib = minimize(
        norm_objective_function,
        norm_initial,
        args=(ocs, ics, sdd, H, param_scales, param_offsets),
        method='L-BFGS-B',
        bounds=norm_bounds,
        options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-6}
    )

    optimised_params = denormalise_params(norm_calib.x, param_scales, param_offsets)
    trans_x, trans_y, trans_z, tilt, roll, initial = optimised_params
    
    if need_plots is True:
        LPRt_matrix = HPRt(optimised_params, sdd, H, num_projs)
        cam = np.matmul(LPRt_matrix, ocs.transpose(2,1,0))
        cam = homo_normalisation(cam.transpose(2,1,0))
        scatter_plot(cam)
        err = np.linalg.norm(cam - ics, axis=1)
        plt.figure(figsize=(6,6))
        plt.imshow(err, aspect='auto')
        plt.colorbar()
        plt.show()
 
    print(f"\nOptimization completed in {norm_calib.nit} iterations")
    print(f"Final error: {norm_calib.fun:.6f}")
    print(f"Success: {norm_calib.success}")
    print("\nOptimized parameters:")
    print(f"trans_x: {trans_x:.4f} mm")
    print(f"trans_y: {trans_y:.2f} mm")
    print(f"sod: {trans_z:.2f} mm")
    print(f"sdd: {sdd:.2f} mm")
    print(f"magnification: {sdd/trans_z:.4f}")
    print(f"tilt: {np.degrees(tilt):.4f} degrees")
    print(f"roll: {np.degrees(roll):.4f} degrees")
    print(f"initial: {np.degrees(initial % (2*np.pi)):.2f} degrees")
    return trans_x, trans_y, trans_z, tilt, roll



def main():
    projs_dir = sys.argv[1]
    
    sdd = float(sys.argv[2])  # mm
    import time, datetime as dt
    t0 = time.perf_counter()
    trans_x, trans_y, trans_z, tilt, roll = calibrate(projs_dir, 
                                                      sdd, 
                                                      need_plots=True)
    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed:.3f} s  ({dt.timedelta(seconds=elapsed)})")
    
    
if  __name__ == "__main__":
    main()


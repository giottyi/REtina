import os
os.environ["OMP_NUM_THREADS"] = '16'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

import glob, sys#, time, datetime as dt


SDD = 690.0  # mm
pxl_size = 61e-3  # mm
eps = np.finfo(np.float16).tiny

linear_camera_matrix = np.array([
    [-1/pxl_size, 0.0, 2750//2],
    [0.0, -1/pxl_size, 2200//2],
    [0.0, 0.0, 1.0]
])

linear_camera_matrix = np.array([
    [-15, 0.0, 2750//2],
    [0.0, -15, 2200//2],
    [0.0, 0.0, 1.0]
])

PARAM_SCALES = np.array([
    100.0,      # trans_x: mm
    100.0,      # trans_y: mm
    1000.0,     # trans_z: mm
    np.pi/4,    # tilt: ±π/4 -> ±1
    np.pi/4,    # roll: ±π/4 -> ±1
    np.pi       # initial: ±π -> ±1
])

PARAM_OFFSETS = np.array([
    0.0,        # trans_x: centered at 0
    0.0,        # trans_y: centered at 0
    600.0,      # trans_z: centered at 600mm
    0.0,        # tilt: centered at 0
    0.0,        # roll: centered at 0
    0.0         # initial: centered at 0
])


def normalize_params(params):
    return (np.array(params) - PARAM_OFFSETS) / PARAM_SCALES


def denormalize_params(norm_params):
    return norm_params * PARAM_SCALES + PARAM_OFFSETS


def normalize_bounds(bounds):
    norm_bounds = []
    for i, (lower, upper) in enumerate(bounds):
        norm_lower = (lower - PARAM_OFFSETS[i]) / PARAM_SCALES[i]
        norm_upper = (upper - PARAM_OFFSETS[i]) / PARAM_SCALES[i]
        norm_bounds.append((norm_lower, norm_upper))
    return norm_bounds


def LPRt(params, num_projs):
    """
    creates the OCS -> WCS -> DCS projection matrices for all views,
    linear model for the post-detector projective geometry, supposing
    known camera matrix. SDD must be know since minimazion only predicts
    magnification and offsets
    """
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

    rot_step = 2.0 * np.pi / (num_projs - 1)
    R_y_fn = lambda n: np.array([ 
        [ np.cos(initial-n*rot_step), 0.0, +np.sin(initial-n*rot_step)],
        [0.0, 1.0, 0.0],
        [-np.sin(initial-n*rot_step), 0.0,  np.cos(initial-n*rot_step)]
    ])
    R_y_projs = np.stack([R_y_fn(n) for n in range(num_projs)])

    R = R_z @ R_x @ R_y_projs
    t = np.array([[trans_x], [trans_y], [trans_z]])
    t = np.tile(t[np.newaxis,:,:], (num_projs,1,1))
    Rt = np.concatenate((R,t), axis=2)
    homo_row = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (num_projs, 1, 1))
    Rt = np.concatenate((Rt, homo_row), axis=1)

    focal_d = SDD
    P = np.array([
        [focal_d, 0.0, 0.0, 0.0],
        [0.0, focal_d, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    L = linear_camera_matrix
    return L @ P @ Rt


def homo_normalization(res):
    res[:,0,:] /= res[:,2,:] + eps
    res[:,1,:] /= res[:,2,:] + eps
    res[:,2,:] = 1.0
    return res


def norm_objective_function(norm_params, ocs, dcs, num_projs):
    """
    normalized for better convergence and stability as in Hartley & Zisserman
    """
    params = denormalize_params(norm_params)
    LPRt_matrix = LPRt(params, num_projs)
    cam = np.matmul(LPRt_matrix, ocs.transpose(2,1,0))
    cam = homo_normalization(cam.transpose(2,1,0))
    return np.linalg.norm(cam-dcs) / num_projs


def adjust_ocs(ocs):
    """
    rotates coordinates from OCS to WCS for consistency and translates
    into baricenter for symmetry
    """
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


def main():
    ocs = adjust_ocs(np.load("../data/phantom_OCS.npy"))
    dcs = np.load("../data/phantom_ICS.npy")
    num_projs = dcs.shape[-1]

    if os.environ.get("VIZ") == "2":
        scatter_plot(dcs)

    initial_params = [
        0.0, 0.0,           # trans_x, trans_y
        600.0,              # trans_z, [mm]
        np.radians(0.0),    # tilt
        np.radians(0.0),    # roll
        np.radians(0.0)     # initial rotation
    ]

    bounds = [
        (-100.0, 100.0),            # trans_x, mm
        (-100.0, 100.0),            # trans_y, mm
        (300.0, 1500.0),            # trans_z, mm
        (-np.pi/4, np.pi/4),        # tilt, radians
        (-np.pi/4, np.pi/4),        # roll, radians
        (-2.0*np.pi, 2.0*np.pi)     # rot, radians
    ]

    if os.environ.get("VIZ") == "2":
        LPRt_matrix = LPRt(initial_params, num_projs)
        cam = np.matmul(LPRt_matrix, ocs.transpose(2,1,0))
        cam = homo_normalization(cam.transpose(2,1,0))
        scatter_plot(cam)
    
    norm_initial = normalize_params(initial_params)
    norm_bounds = normalize_bounds(bounds)
    norm_calib = minimize(
        norm_objective_function,
        norm_initial,
        args=(ocs, dcs, num_projs),
        method='L-BFGS-B',
        bounds=norm_bounds,
        options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-6}
    )

    optimized_params = denormalize_params(norm_calib.x)
    print(f"\nOptimization completed in {norm_calib.nit} iterations")
    print(f"Final error: {norm_calib.fun:.6f}")
    print(f"Success: {norm_calib.success}")
    print("\nOptimized parameters:")
    print(f"trans_x: {optimized_params[0]:.2f} mm")
    print(f"trans_y: {optimized_params[1]:.2f} mm")
    print(f"SOD: {optimized_params[2]:.2f} mm")
    print(f"SDD: {SDD:.2f} mm")
    print(f"magnification: {SDD/optimized_params[2]:.3f}")
    print(f"tilt: {np.degrees(optimized_params[3]):.2f} degrees")
    print(f"roll: {np.degrees(optimized_params[4]):.2f} degrees")
    print(f"initial: {np.degrees(optimized_params[5] % (2*np.pi)):.2f} degrees")
    
    if os.environ.get("VIZ") == "2":
        LPRt_matrix = LPRt(optimized_params, num_projs)
        cam = np.matmul(LPRt_matrix, ocs.transpose(2,1,0))
        cam = homo_normalization(cam.transpose(2,1,0))
        scatter_plot(cam)


if __name__ == "__main__":
    main()


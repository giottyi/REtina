
import os
os.environ["OMP_NUM_THREADS"] = '16'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import astra
#astra.test_CUDA()

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.signal import periodogram

import threading, time, sys, datetime as dt

from preprocess import get_data, convert_u16, add_noise
from calibrate import calibrate


def spinner(msg, stop_event):
    ''' instead of print(f'Running {algo} ...', end='', flush=True) '''
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f'\r{msg} {spinner_chars[idx % len(spinner_chars)]}')
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    ok_msg = 'OK'
    sys.stdout.write(f'\r{msg} {ok_msg}\n')


def _stop_iterative(sinogram, vol_geom, proj_geom, algo_id, recon_id, max_iter, patience=2):
  height, num_projs, width = sinogram.shape

  ncp_history = np.inf * np.ones(max_iter)
  pgram_length = width // 2 + 1
  white_periodogram = np.linspace(1/pgram_length, 1, pgram_length, endpoint=True)
  count = 0

  astra.set_gpu_index(0)
  for i in range(max_iter):
    astra.algorithm.run(algo_id, iterations=1)
    recon = astra.data3d.get(recon_id)
    astra.data3d.store(recon_id, recon)
    vol_id = astra.data3d.create('-vol', vol_geom, recon)
    proj_id, proj_est = astra.creators.create_sino3d_gpu(vol_id, proj_geom, vol_geom)
    residual = sinogram - proj_est

    pxx = np.empty((num_projs, pgram_length))
    for proj in range(num_projs):
      pxx[proj] = periodogram(residual[height // 2, proj])[1]
    ncp = np.cumsum(pxx, axis=-1) / np.sum(pxx, axis=-1, keepdims=True)
    ncp_diff = np.linalg.norm(ncp - white_periodogram, ord=1)
    if ncp_diff <= ncp_history.min():
      ncp_iter = i
      ncp_recon = recon
      count = 0
    else:
      count += 1
    ncp_history[i] = ncp_diff

    if count == patience:
      break
  return ncp_history, ncp_iter, ncp_recon


def _roto_translate_geom(projection_geom, tilt_angle=0.0, roll_angle=0.0, delta_x=0.0, delta_y=0.0):
  G = astra.geom_2vec(projection_geom).copy()

  S = np.copy(G['Vectors'][0,0:3])   # source → object
  D = np.copy(G['Vectors'][0,3:6])   # object → detector center
  u = np.copy(G['Vectors'][0,6:9])   # detector u-axis
  v = np.copy(G['Vectors'][0,9:])    # detector v-axis

  # Rotation of the tomography (around y-axis)
  angles = projection_geom['ProjectionAngles']
  R_tomo = np.array([
    [[np.cos(a), -np.sin(a), 0.0],
      [np.sin(a),  np.cos(a), 0.0],
      [0.0, 0.0, 1.0]] for a in angles])

  # Roll (around z) and tilt (around x)
  R_roll = np.array([
    [np.cos(roll_angle), 0.0, -np.sin(roll_angle)],
    [0.0, 1.0, 0.0],
    [np.sin(roll_angle), 0.0,  np.cos(roll_angle)]
  ])
  R_tilt = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(tilt_angle), -np.sin(tilt_angle)],
    [0.0, np.sin(tilt_angle),  np.cos(tilt_angle)]
  ])
  R_global = R_roll @ R_tilt  # Apply tilt first, then roll

  # --- Apply global rotation ---
  S_rot = R_global @ S
  D_rot = R_global @ D
  u_rot = R_global @ u
  v_rot = R_global @ v

  # --- Then apply detector translation (in rotated local frame) ---
  D_shifted = D_rot + delta_x * u_rot + delta_y * v_rot

  # --- Apply per-angle tomography rotation ---
  n_proj = len(angles)
  G_new = np.zeros_like(G['Vectors'])
  G_new[:,0:3] = (R_tomo @ S_rot).reshape(n_proj, 3)
  G_new[:,3:6] = (R_tomo @ D_shifted).reshape(n_proj, 3)
  G_new[:,6:9] = (R_tomo @ u_rot).reshape(n_proj, 3)
  G_new[:,9:12] = (R_tomo @ v_rot).reshape(n_proj, 3)
  G['Vectors'] = G_new
  return G


def create_proj_geom(sinogram, angles, sdd, sod, pxl_size=61e-3, tilt=0.0, roll=0.0, shift_x=0.0, shift_y=0.0):
  M = sdd / sod
  vxl_size = pxl_size / M  # mm
  det_rows, num_projs, det_cols = sinogram.shape
  proj_geom = astra.create_proj_geom('cone', 
                                     M, 
                                     M, 
                                     det_rows, 
                                     det_cols, 
                                     angles, 
                                     sod/vxl_size, 
                                     (sdd-sod)/vxl_size)
  return _roto_translate_geom(proj_geom, tilt, roll, shift_x/vxl_size, shift_y/vxl_size)


def reconstruct(sinogram, proj_geom, algo='FDK_CUDA', max_iter=10, prev_ncp_iter=None):
  ''' lengths must be expressed in number of voxels
  '''
  assert algo in ['FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA'], "Algorithm not valid"

  det_rows, num_projs, det_cols = sinogram.shape
  vol_geom = astra.create_vol_geom(det_cols, det_cols, det_rows)
  recon_id = astra.data3d.create('-vol', vol_geom)
  proj_id = astra.data3d.create('-proj3d', proj_geom, sinogram)

  astra.set_gpu_index(0)
  recon_id = astra.data3d.create('-vol', vol_geom)
  cfg = astra.astra_dict(algo)
  cfg['ReconstructionDataId'] = recon_id
  cfg['ProjectionDataId'] = proj_id
  algo_id = astra.algorithm.create(cfg)

  stop_spinner = threading.Event()
  spinner_thread = threading.Thread(target=spinner, 
                                    args=(f'Running {algo} ...', stop_spinner))
  spinner_thread.start()
  if algo == 'FDK_CUDA':
    astra.algorithm.run(algo_id)
    recon = astra.data3d.get(recon_id)
  elif prev_ncp_iter is not None:
    astra.algorithm.run(algo_id, iterations=prev_ncp_iter)
    recon = astra.data3d.get(recon_id)
  else:
    ncp_history, ncp_iter, recon = _stop_iterative(sinogram, 
                                                   vol_geom, 
                                                   proj_geom, 
                                                   algo_id, 
                                                   recon_id, 
                                                   max_iter)
 
  stop_spinner.set()
  spinner_thread.join()
  astra.functions.clear()

  if algo == 'FDK_CUDA':
    return None, None, recon
  elif prev_ncp_iter is not None:
    return None, None, recon
  else:
    print(f'Stopped @ iteration : {ncp_iter+1}')
    if ncp_iter + 1 == max_iter:
      print('Try increasing maximum number of iterations')
    return ncp_history, ncp_iter, recon
  


def main():
  projs_dir = r'E:\RETINA\rawdata\pomegranade_101125\projections\70kV_2.82mA_2500ms_binning2'
  flats_dir = r'E:\RETINA\rawdata\pomegranade_101125\flats\70kV_2.82mA_2500ms_binning2'
  darks_dir = r'E:\RETINA\rawdata\pomegranade_101125\darks\70kV_2.82mA_2500ms_binning2'
  phantom_projs_dir = r'E:\RETINA\rawdata\pomegranade_101125\phantom\projections'

  sdd = 572.0  # mm
  need_calib = False
  need_data = False
  path_input = '../recon_data/'
  file_input = 'prova'
  file_output = 'prova_fdk'

  crop = (850,850) # (rows, cols) or None
  binning = 1 # 1, 2, 3, ...
  pxl_size = binning*62e-3  # mm

  if need_calib is True: 
    trans_x, trans_y, trans_z, tilt, roll = calibrate(phantom_projs_dir, 
                                                      None,  
                                                      sdd, 
                                                      need_plots=True)
  else:
    trans_x, trans_y, trans_z, tilt, roll = 3.5276, -11.96, 541.72, np.deg2rad(-0.4011), np.deg2rad(1.1357) # nut

  trans_x += -0.2 # characterized for RETINA
  trans_y *= 0
  tilt *= -1
  roll *= +1

  geom_params = {'sdd':sdd, 
                 'sod':trans_z, 
                 'pxl_size':pxl_size*binning, 
                 'tilt':tilt, 
                 'roll':roll, 
                 'shift_x':trans_x, 
                 'shift_y':trans_y}

  if need_data is True:
    angles, sinogram = get_data(projs_dir, 
                                flats_dir, 
                                darks_dir, 
                                need_angles=True, 
                                crop=crop, 
                                binning=binning, 
                                endpoint=False, 
                                remove_outliers=True)
  else:
    sinogram = np.load(f'{path_input}{file_input}.npy')
    pass
  angles = np.linspace(0, 2*np.pi, sinogram.shape[1], endpoint=False)

  algo_type = 'FDK_CUDA'
  max_iter = 30
  proj_geom = create_proj_geom(sinogram, angles, **geom_params)
  _, _, recon = reconstruct(sinogram, 
                            proj_geom, 
                            algo_type, 
                            max_iter)
  
  np.save(f'../n2i_data/{file_output}.npy', recon)
  tifffile.imwrite(f'../zz_output/{file_output}.tiff', convert_u16(recon))


if __name__ == "__main__":
  main()


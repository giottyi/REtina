import os
os.environ["OMP_NUM_THREADS"] = '16'

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from astropy.io import fits
from tqdm import tqdm

import glob, sys#, time, datetime as dt


#flats_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\flats'
#darks_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\darks'
#projs_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\\trasfomatore\Trasformatore_110kV_4.6mA_2s\projections'

crop = None


def get_views(directory, crop=None, fit=True):
    if fit is True: 
        views_paths = sorted(glob.glob(os.path.join(directory, "*.fit")))
        if not views_paths:
            raise UserWarning("No FIT files found in directory.")

        hdul = fits.open(views_paths[0])
        sample_view = hdul[0].data

    else: 
        views_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
        if not views_paths:
            raise UserWarning("No TIFF files found in directory.")

        sample_view = imread(views_paths[0])

    if crop is not None:
        crop_width, crop_height = crop
        height, width = sample_view.shape
        cx, cy = width // 2, height // 2
        x1 = cx - crop_width // 2
        x2 = x1 + crop_width
        y1 = cy - crop_height // 2
        y2 = y1 + crop_height
        sample_view = np.flipud(sample_view)[y1:y2, x1:x2]
    else: sample_view = np.flipud(sample_view)
    n_views = len(views_paths)
    views_stack = np.empty((n_views, *sample_view.shape), dtype=sample_view.dtype)
    views_angles = np.empty(n_views, dtype=int)

    views_stack[0] = sample_view
    #views_angles[0] = int(views_paths[0].split('.')[0].split('_')[-1])
    for i, view_path in enumerate(tqdm(views_paths[1:], desc="Loading views"), start=1):
        if fit is True: 
            hdul = fits.open(view_path)
            view = hdul[0].data
        else: view = imread(view_path)
        if crop is not None:
            views_stack[i] = np.flipud(view)[y1:y2, x1:x2]
        else: views_stack[i] = np.flipud(view)
        #views_angles[i] = int(view_path.split('.')[0].split('_')[-1])
    return views_angles, views_stack


def get_flat(directory, crop=None, fit=True):
    _, flats_stack = get_views(directory, crop=crop, fit=fit)
    return np.median(flats_stack, axis=0).astype(flats_stack.dtype)


def get_dark(directory, crop=None, fit=True):
    return get_flat(directory, crop=crop, fit=fit)


def get_data(views_path, flat_path, dark_path=None, crop=None, fit=True):
    """
    applies flat-dark correction and minus_log
    """
    flat = get_flat(flat_path, crop=crop, fit=fit)
    angles, views = get_views(views_path, crop=crop, fit=fit)
    if dark_path is None:
        dark = np.zeros_like(flat)
    else: dark = get_dark(dark_path, crop=crop, fit=fit)

    def correct_log(views, flat, dark, dtype=np.float32, chunk=128):
        eps = np.finfo(np.float16).tiny
        denom = (flat - dark).astype(dtype, copy=False)
        out = np.empty(views.shape, dtype=dtype)
        for i in range(0, views.shape[0], chunk):
            sl = slice(i, i+chunk)
            tmp = views[sl].astype(dtype, copy=False)
            np.subtract(tmp, dark, out=tmp)
            np.add(denom, eps, out=denom)
            np.maximum(tmp, eps, out=tmp)
            np.divide(tmp, denom, out=tmp)
            np.log(tmp, out=tmp)
            out[sl] = -tmp
        return out
    return angles, correct_log(views, flat, dark)


def main():
    """
    currently crops dataset to reconstruct with astra
    """
    #t0 = time.perf_counter() 
    #angles, data = get_data(projs_dir, flats_dir, darks_dir, crop=None)
    #print(data.dtype, data.min(), data.max())
    #print(data.shape)
    #elapsed = time.perf_counter() - t0
    #print(f"Finished in {elapsed:.3f} s  ({dt.timedelta(seconds=elapsed)})")
    


if __name__ == "__main__":
    main()



import os
os.environ["OMP_NUM_THREADS"] = '16'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy.ndimage import median_filter
from tifffile import imread, imwrite
from tqdm import tqdm

import glob, re, time, datetime as dt


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _load_view(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"): 
        return np.flipud(imread(path))
    elif ext in (".fit", ".fits", ".fts"): 
        from astropy.io import fits
        with fits.open(path) as fit:
            return np.flipud(fit[0].data)
    else: 
        raise ValueError(f"Unsupported file type: {ext}")
    

def _rebin_2d(views, y_bin=2, x_bin=2, op='mean'):
    num_projs, height, width = views.shape
    height_binned = (height // y_bin) * y_bin
    width_binned = (width // x_bin) * x_bin
    views = views[:, :height_binned, :width_binned]
    views = views.reshape(num_projs, 
                          height_binned // y_bin, y_bin, 
                          width_binned // x_bin, x_bin)
    views = views.swapaxes(2, 3)

    if op == 'mean':
        views = views.mean(axis=(3, 4))
    elif op == 'sum':
        views = views.sum(axis=(3, 4))
    else:
        raise ValueError("Binning operation must be 'mean' or 'sum'")
    return views


def convert_u16(recon, chunk=128):
  eps = np.finfo(np.float16).tiny
  recon_min = recon.min().astype(recon.dtype)
  recon_max = recon.max().astype(recon.dtype)
  denom = (recon_max - recon_min).astype(recon.dtype)
  denom += eps
  out = np.empty(recon.shape, dtype=np.uint16)
  for i in range(0, recon.shape[0], chunk):
    sl = slice(i, i+chunk)
    tmp = recon[sl].astype(recon.dtype, copy=False)
    np.subtract(tmp, recon_min, out=tmp)
    np.maximum(tmp, 0, out=tmp)
    np.divide(tmp, denom, out=tmp)
    np.multiply(tmp, 65535, out=tmp)
    out[sl] = tmp.astype(np.uint16)
  return out


def add_noise(sino, count=1000, seed=0, chunk=128): 
    out = np.empty(sino.shape, dtype=sino.dtype)
    rng = np.random.default_rng(seed) 
    for i in range(0, sino.shape[0], chunk): 
        sl = slice(i, i+chunk) 
        tmp = sino[sl].astype(sino.dtype, copy=False) 
        np.negative(tmp, out=tmp) 
        np.exp(tmp, out=tmp) 
        np.multiply(tmp, count, out=tmp) 
        tmp = rng.poisson(tmp).astype(sino.dtype) 
        tmp[tmp == 0] = 1.0 
        np.divide(tmp, count, out=tmp) 
        np.log(tmp, out=tmp) 
        np.negative(tmp, out=tmp) 
        out[sl] = tmp.astype(sino.dtype, copy=False) 
    return out


def get_views(directory, need_angles=False, crop=None, binning=1, views_type='views'):
    views_paths = sorted(
        glob.glob(os.path.join(directory, "*.*")),
        key=_natural_key
    )
    views_paths = [p for p in views_paths if os.path.splitext(p)[1].lower() in \
            (".fit", ".fits", ".fts", ".tif", ".tiff")]
    if not views_paths: 
        raise ValueError("No FITS or TIFF files found in directory.")

    sample_view = _load_view(views_paths[0])

    if crop is not None:
        crop_height, crop_width = crop
        height, width = sample_view.shape
        cx, cy = width // 2, height // 2
        x1 = cx - crop_width // 2
        x2 = x1 + crop_width
        y1 = cy - crop_height // 2
        y2 = y1 + crop_height
        sample_view = sample_view[y1:y2,x1:x2]

    n_views = len(views_paths)
    views_stack = np.empty((n_views, *sample_view.shape), dtype=sample_view.dtype)
    views_stack[0] = sample_view
    if need_angles is True:
        views_angles = np.empty(n_views, dtype=int)
        views_angles[0] = int(views_paths[0].split('_')[-1].split('.')[0])

    for i, view_path in enumerate(tqdm(views_paths[1:], 
                                       desc=f"Loading {views_type}", 
                                       total=n_views, initial=1), start=1):
        view = _load_view(view_path)
        if crop is not None:
            view = view[y1:y2,x1:x2]
        views_stack[i] = view
        if need_angles is True:
            views_angles[i] = int(view_path.split('_')[-1].split('.')[0])

    if binning > 1:
        views_stack = _rebin_2d(views_stack, y_bin=binning, x_bin=binning)

    if need_angles is True:
        return np.deg2rad(views_angles), views_stack
    else: 
        return None, views_stack


def get_flat(directory, crop=None, binning=1):
    _, flats_stack = get_views(directory, 
                               need_angles=False, 
                               crop=crop, 
                               binning=binning, 
                               views_type='flats')
    return np.median(flats_stack, axis=0).astype(flats_stack.dtype)


def get_dark(directory, crop=None, binning=1):
    _, darks_stack = get_views(directory, 
                               need_angles=False, 
                               crop=crop, 
                               binning=binning, 
                               views_type='darks')
    return np.median(darks_stack, axis=0).astype(darks_stack.dtype)
   


def get_data(views_path, flats_path, darks_path=None, need_angles=False, crop=None, binning=1, endpoint=False, remove_outliers=False):
    """ applies flat_dark correction and minus_log
    """
    flat = get_flat(flats_path, crop=crop, binning=binning)
    angles, views = get_views(views_path, 
                              need_angles=need_angles, 
                              crop=crop, 
                              binning=binning, 
                              views_type='projections')
    if darks_path is None:
        dark = np.zeros_like(flat)
    else: 
        dark = get_dark(darks_path, crop=crop, binning=binning)

    def _correct_log(views, flat, dark, dtype=np.float32, chunk=128):
        ''' experimental'''
        if remove_outliers is True:
            k_sz = 3
            median_filter(views, size=(1,k_sz,k_sz), output=views, mode='nearest')

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
        return out.transpose(1,0,2)
        
    if endpoint is True:
        return angles, _correct_log(views, flat, dark)
    elif endpoint is False and angles is None:
        return None, _correct_log(views, flat, dark)[:,:-1,:]
    elif endpoint is False and angles is not None: 
        return angles[:-1], _correct_log(views, flat, dark)[:,:-1,:]



def main():
    projs_dir = r'/home/francesco/Desktop/tesi/phantom_calib/2025_05_16/690SDD_506SOD'
    flats_dir = r'/home/francesco/Desktop/tesi/phantom_calib/2025_05_16/690SDD_flats/506SOD'
    darks_dir = None

    crop = (2000, 2400) # (rows, cols) or None
    binning = 4  # 2, 3, ...

    t0 = time.perf_counter() 
    _, sino = get_data(projs_dir, 
                        flats_dir, 
                        darks_dir, 
                        need_angles=False, 
                        crop=crop, 
                        binning=binning, 
                        endpoint=False, 
                        remove_outliers=True)
    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed:.3f} s  ({dt.timedelta(seconds=elapsed)})")

    path = '../recon_data/'
    filename = 'prova'

    np.save(f'{path}{filename}.npy', sino)
    imwrite(f'{path}{filename}.tiff', convert_u16(sino))



if __name__ == "__main__":
    main()


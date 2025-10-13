import os
os.environ["OMP_NUM_THREADS"] = '16'

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from astropy.io import fits
import cv2
from tqdm import tqdm

import glob, re, sys#, time, datetime as dt


#flats_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\flats'
#darks_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\darks'
#projs_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\\trasfomatore\Trasformatore_110kV_4.6mA_2s\projections'

crop = None


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _load_view(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".fits", ".fts"):
        with fits.open(path) as fit:
            return np.flipud(fit[0].data)
    elif ext in (".tif", ".tiff"):
        return np.flipud(imread(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_views(directory, crop=None):
    views_paths = sorted(
        glob.glob(os.path.join(directory, "*.*")),
        key=_natural_key
    )
    views_paths = [p for p in views_paths if os.path.splitext(p)[1].lower() in \
            (".fit", ".fits", ".fts", ".tif", ".tiff")]
    if not views_paths:
        raise UserWarning("No FITS or TIFF files found in directory.")

    sample_view = _load_view(views_paths[0])

    if crop is not None:
        crop_width, crop_height = crop
        height, width = sample_view.shape
        cx, cy = width // 2, height // 2
        x1 = cx - crop_width // 2
        x2 = x1 + crop_width
        y1 = cy - crop_height // 2
        y2 = y1 + crop_height
        sample_view = sample_view[y1:y2, x1:x2]

    n_views = len(views_paths)
    views_stack = np.empty((n_views, *sample_view.shape), dtype=sample_view.dtype)
    views_angles = np.empty(n_views, dtype=int)

    views_stack[0] = sample_view
    #views_angles[0] = int(views_paths[0].split('.')[0].split('_')[-1])
    for i, view_path in enumerate(tqdm(views_paths[1:], desc="Loading views", \
            total=n_views, initial=1), start=1):
        view = _load_view(view_path)
        if crop is not None:
            view = view[y1:y2, x1:x2]
        views_stack[i] = view
        #views_angles[i] = int(view_path.split('.')[0].split('_')[-1])
    return views_angles, views_stack


def get_flat(directory, crop=None):
    _, flats_stack = get_views(directory, crop=crop)
    return np.median(flats_stack, axis=0).astype(flats_stack.dtype)


def get_dark(directory, crop=None):
    return get_flat(directory, crop=crop)


def get_data(views_path, flat_path, dark_path=None, crop=None, remove_outliers=True):
    """
    applies flat_dark correction and minus_log
    """
    flat = get_flat(flat_path, crop=crop)
    angles, views = get_views(views_path, crop=crop)
    if dark_path is None:
        dark = np.zeros_like(flat)
    else: dark = get_dark(dark_path, crop=crop)

    def _correct_log(views, flat, dark, dtype=np.float32, chunk=128):
        eps = np.finfo(np.float16).tiny
        denom = (flat - dark).astype(dtype, copy=False)
        out = np.empty(views.shape, dtype=dtype)
        if remove_outliers == True:
            # experimental
            for i, view in enumerate(views):
                views[i] = cv2.bilateralFilter(view.astype(dtype, copy=False),
                                               d=9, sigmaColor=75, sigmaSpace=75)

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
    return angles, _correct_log(views, flat, dark)


def main():
    """
    currently crops dataset to reconstruct with astra
    """
    #t0 = time.perf_counter() 
    projs_dir, flats_dir = sys.argv[1:]
    angles, data = get_data(projs_dir, flats_dir, crop=None)
    #print(data.dtype, data.min(), data.max())
    #print(data.shape)
    #elapsed = time.perf_counter() - t0
    #print(f"Finished in {elapsed:.3f} s  ({dt.timedelta(seconds=elapsed)})")

    plt.imshow(data[1], cmap='gray')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()
    


if __name__ == "__main__":
    main()


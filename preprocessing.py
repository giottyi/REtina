import os
os.environ["OMP_NUM_THREADS"] = '16'

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from tqdm import tqdm

import glob, time, datetime as dt


flats_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\flats'
darks_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\darks'
projs_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\\trasfomatore\Trasformatore_110kV_4.6mA_2s\projections'


def get_views(directory):
    views_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
    if not views_paths:
        raise UserWarning("No TIFF files found in directory.")

    sample_view = imread(views_paths[0])
    sample_view = np.flipud(sample_view)
    height, width = sample_view.shape
    n_views = len(views_paths)
    views_stack = np.empty((n_views, height, width), dtype=sample_view.dtype)
    views_angles = np.empty(n_views, dtype=int)

    views_stack[0] = sample_view
    views_angles[0] = int(views_paths[0].split('.')[1].split('_')[2])
    for i, view_path in enumerate(tqdm(views_paths[1:], desc="Loading views"), start=1):
        view = imread(view_path)
        views_stack[i] = np.flipud(view)
        views_angles[i] = int(view_path.split('.')[1].split('_')[2])
    return views_angles, views_stack


def get_flat(directory):
    """
    averages flats maintaining precision, used in minus_log correction
    """
    _, flats_stack = get_views(directory)
    return np.median(flats_stack, axis=0).astype(flats_stack.dtype)


def get_dark(directory):
    """
    averages darks maintaining precision, used in flat_dark correction
    """
    return get_flat(directory)


def get_data(views_path, flat_path, dark_path=None):
    """
    applies flat-dark correction and minus_log
    """
    flat = get_flat(flat_path)
    angles, views = get_views(views_path)
    if dark_path is None:
        dark = np.zeros_like(flat)
    else:
        dark = get_dark(dark_path)

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
    t0 = time.perf_counter() 
    angles, data = get_data(projs_dir, flats_dir, darks_dir)
    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed:.3f} s  ({dt.timedelta(seconds=elapsed)})")

    plt.imshow(data[0], cmap='seismic')
    plt.colorbar()
    plt.tight_layout()
    plt.gca().invert_yaxis()
    #plt.savefig("./sample_projection.png", dpi=300)



if __name__ == "__main__":
    main()


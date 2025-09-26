import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from tqdm import tqdm

import glob, os, sys


flats_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\flats'
darks_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\trasfomatore\Trasformatore_110kV_4.6mA_2s\darks'
projs_dir = r'E:\OneDrive - Politecnico di Milano\NDT@DENG\RETINA\Users\Leone Di Lernia\\trasfomatore\Trasformatore_110kV_4.6mA_2s\projections'


def get_views(directory, crop=None):
    views_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
    if not views_paths:
        raise ValueError("No TIFF files found in directory.")

    sample_view = imread(views_paths[0])
    sample_view = np.flipud(sample_view)
    if sample_view is None:
        raise ValueError(f"Failed to read sample image: {views_paths[0]}")

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


def main():
    """
    currently crops dataset to reconstruct with astra
    """
    #flat = get_flat(flats_dir)
    #dark = get_dark(darks_dir)
    _, data = get_views(projs_dir)

    plt.imshow(data[0], cmap='gray')
    plt.gca().invert_yaxis() 
    plt.savefig('raw_projection.png')



if __name__ == "__main__":
    main()


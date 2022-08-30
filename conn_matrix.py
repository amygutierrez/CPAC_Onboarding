import os
import argparse
import numpy as np
from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_label_regions

parser = argparse.ArgumentParser()
parser.add_argument('--timeseries', '-ts', type = str, help = 'functional time series data')
parser.add_argument('--mask', '-m', type = str, help = 'functional brain mask used for NiftiMasker')
parser.add_argument('--atlas', '-a', type = str, help = 'atlas used for NiftiLabelMasker')
args = parser.parse_args()

desc_preproc = args.timeseries
atlas = args.atlas
mask = args.mask


if atlas is not None:
    base = os.path.basename(atlas)
    atlas_name = os.path.splitext(base)[0]
    print(f'* * Plotting connecitivty matrix with atlas {atlas_name} * *')
    region_labels = connected_label_regions(atlas) 
    masker = NiftiLabelsMasker(labels_img = region_labels, standardize=True, memory='nilearn_cache', verbose=5)
    timeseries_cpac = masker.fit_transform(desc_preproc)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix_cpac = correlation_measure.fit_transform([timeseries_cpac])[0]
    print(f'Connectivity matrix shape: {correlation_matrix_cpac.shape}')

    np.fill_diagonal(correlation_matrix_cpac, 0)
    plotting.plot_matrix(correlation_matrix_cpac, figure=(10, 10),
                        vmin=-0.4, vmax = 0.6)
    plotting.show()

if mask is not None:
    base = os.path.basename(mask)
    mask_name = os.path.splitext(base)[0]
    print(f'* * Plotting connecitivty matrix with mask {mask_name} * *')
    masker = NiftiMasker(mask_img = mask, standardize=True, memory='nilearn_cache', verbose=5, memory_level=2) 
    timeseries_cpac = masker.fit_transform(desc_preproc)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix_cpac = correlation_measure.fit_transform([timeseries_cpac])[0]
    print(f'Connectivity matrix shape: {correlation_matrix_cpac.shape}')

    np.fill_diagonal(correlation_matrix_cpac, 0)
    plotting.plot_matrix(correlation_matrix_cpac, figure=(10, 10),
                        vmin=-0.4, vmax = 0.6)
    plotting.show()

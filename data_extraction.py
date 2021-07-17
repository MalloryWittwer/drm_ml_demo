'''
@ Mallory Wittwer (mallory.wittwer@ntu.edu.sg), May 2021
@ Matteo Seita (mseita@ntu.edu.sg)

Demonstration of the extraction of training and evaluation sets based on 
grain segmentation by the LRC-MRM method.
'''
import os
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from lib.segmentation import run_lrc_mrm

def run_test(ms=8, cps=20, ssize=5_000, debug=False, save_it=True):
    '''
    ms: Exclusion distance to closest grain boundary (in px).
    cps: Size of the NMF basis (pseudo-parameter of the LRC-MRM algorithm)
    ssize: Random sampling size (pseudo-parameter of LRC-MRM)
    debug: stops the process after 1 window
    save_it: whether to save the extracted dataset or not
    '''
    # Specify root directory
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(os.path.dirname(root), 'data/'))
    
    # Open DRM dataset
    data = np.load(f'{root}/drm_data.npy')
    rx, ry, s0, s1 = data.shape
    
    # Open EBSD euler map (labels for training / evaluation)
    eulers = np.load(f'{root}/eulers.npy')
    
    def data_window_generator():
        '''
        Yields a sliding window of (200 x 200) pixels across the dataset.
        '''
        window_size_x, window_size_y = (200,200)
        x_indeces = np.floor(rx//window_size_x)
        residual_x = rx - window_size_x*x_indeces
        y_indeces = np.floor(ry//window_size_y)
        residual_y = ry - window_size_y*y_indeces
        for kx in range(int(x_indeces)):
            xstart = kx*window_size_x
            xend = xstart + window_size_x if xstart + window_size_x <= rx else xstart + residual_x
            for ky in range(int(y_indeces)):
                ystart = ky*window_size_y
                yend = ystart + window_size_y if ystart + window_size_y <= ry else ystart + residual_y
                data_slice = data[xstart:xend, ystart:yend]
                eulers_slice = eulers[xstart:xend, ystart:yend]
                yield data_slice, eulers_slice
    
    xtr = np.empty((0, s0, s1))
    ytr = np.empty((0, 3))
    
    # Process the entire dataset in sliding windows to avoid memory overflow
    for data_slice, eulers_slice in data_window_generator():
        
        # Size of the selected window
        rxx, ryy, *_ = data_slice.shape
        
        # Create input dataset to LRC-MRM segmentaiton algorithm
        dataset_slice = {
            'data':data_slice.reshape((rxx*ryy, s0*s1)), 
            'eulers':eulers_slice.reshape((rxx*ryy, 3)),
            'spatial_resol':(rxx,ryy), 
            'angular_resol':(s0,s1),
            }

        # Run segmentation algorithm
        dataset_slice = run_lrc_mrm(dataset_slice, cps, ssize)
        
        # Get grain segmentation map
        segmentation = dataset_slice.get('segmentation').reshape((rxx, ryy))
        
        # Get grain boundary map
        gbs = dataset_slice.get('boundaries').reshape((rxx, ryy))
        
        # Euclidean Distance Transform
        distance = ndimage.distance_transform_edt(~gbs)
        
        # Select data at maximum of EDT (per grain)
        local_maxi = peak_local_max(
            distance, # provide the EDT
            labels=segmentation, # provide the grain map
            num_peaks_per_label=1, # 1 point per grain
            threshold_abs=ms, # minimum distance = 'ms' pixels away
            indices=False, 
        )
        
        # If at least 1 point is found, add selection to the total
        if local_maxi.sum()>1:
            ds = data_slice[local_maxi]
            es = eulers_slice[local_maxi]
            xtr = np.vstack((xtr, ds))
            ytr = np.vstack((ytr, es))
                
        if debug:
            break # Break the loop in debug mode

    # Save the completed dataset
    if save_it:
        np.save(f'{root}/extracted_set_test.npy', {'xtr':xtr, 'ytr':ytr})
    
if __name__ == '__main__':
    '''
    Expected result: Creation of NPY file (extracted_set_test.npy) in the root
    directory. The file contains a dataset of (1) DRM signals extracted from
    the provided specimen and (2) the corresponding EBSD crystal orientations.
    This strategy was applied to 10 different specimens before training the 
    CNN models shown in the paper.
    '''
    run_test(debug=True, save_it=True)
'''
Utility functions.
'''

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import tensorflow as tf

def array_trimmer(bs, *arrays):
    trimmed_arrays = []
    for a in arrays:
        trimmed_arrays.append(a[:len(a)-len(a)%bs])
    return trimmed_arrays

def timeit(method):
    '''
    Timer decorator - measures the running time of a specific function.
    '''
    def timed(*args, **kw):
        print('\n> Starting: {} \n'.format(method.__name__))
        # ts = time.time()
        ts = perf_counter()
        result = method(*args, **kw)
        te = perf_counter()
        # te = time.time()
        print('\n> Timer ({}): {:.2f} sec.'.format(method.__name__, te-ts))
        return result
    return timed

def show(im, cmap=plt.cm.jet, s=12, vmin=None, vmax=None, title=''):
    '''A basic display of the image.'''
    fig, ax = plt.subplots(figsize=(s,s))
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)
    plt.show()
    
def splitter(data, sample_size, test_fraction):
    '''
    Splits the data according to a fixed sample size. Separates the sampled
    data into a training and test fraction. Returns the sampled data with the 
    indeces of the training and test samples.
    '''
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data_extract = data[idx[:sample_size]]
    train_size = np.clip(sample_size*(1-test_fraction), 0, sample_size)
    train_size = train_size.astype('int')
    train_sample = data_extract[:train_size]
    test_sample = data_extract[train_size:sample_size]
    idxtr = idx[:train_size]
    idxte = idx[train_size:sample_size]
    return train_sample, idxtr, test_sample, idxte

def shuffler(data, sample_size):
    '''
    Randomly selects a sample of sample_size from the data and returns it 
    with the corresponding indeces.
    '''
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    sample = data[idx[:sample_size]]
    indeces = idx[:sample_size]
    return sample, indeces

def get_xymaps(rx, ry):
    '''
    Produces a flattened mesh grid of X and Y coordinates of resolution (rx,ry)
    '''
    # X-map
    xmap = np.empty((rx*ry))
    for k in range(rx):
        xmap[k*ry:(k+1)*ry] = np.array([k]*ry)
    xmap = xmap.reshape((rx, ry))
    xmap = xmap.ravel()
    # Y-map
    ymap = np.empty((rx*ry))
    for k in range(rx):
        ymap[k*ry:(k+1)*ry] = np.arange(ry)
    ymap = ymap.reshape((rx, ry))
    ymap = ymap.ravel()
    return xmap, ymap

def png_to_eulers(eulers):
    '''Converts an euler matrix from an image back to angular units'''
    eulers = eulers.astype(np.float32)
    eulers[:,:,0] = eulers[:,:,0]/255*np.pi*2
    eulers[:,:,1] = eulers[:,:,1]/255*np.pi/2
    eulers[:,:,2] = eulers[:,:,2]/255*np.pi/2
    return eulers

def eulers_to_png(eulers):
    '''Converts euler angles (radians) to an image (0-255)'''
    eulers[...,0] /= np.pi*2
    eulers[...,1] /= np.pi/2
    eulers[...,2] /= np.pi/2
    eulers = (eulers*255).astype(np.uint8)
    return eulers

def eulers_to_rot_mat(eulers):
    i1  = eulers[:,0]
    i2  = eulers[:,1]
    i3  = eulers[:,2]       
    i1c = tf.cos(i1)
    i1s = tf.sin(i1)
    i2c = tf.cos(i2)
    i2s = tf.sin(i2)
    i3c = tf.cos(i3)
    i3s = tf.sin(i3)
    x00 = i1c*i2c*i3c-i1s*i3s
    x01 = -i3c*i1s-i1c*i2c*i3s
    x02 = i1c*i2s
    x10 = i1c*i3s+i2c*i3c*i1s
    x11 = i1c*i3c-i2c*i1s*i3s
    x12 = i1s*i2s
    x20 = -i3c*i2s
    x21 = i2s*i3s
    x22 = i2c
    c0 = tf.stack((x00,x01,x02), axis=1)
    c1 = tf.stack((x10,x11,x12), axis=1)
    c2 = tf.stack((x20,x21,x22), axis=1)
    rot_mat = tf.stack((c0,c1,c2), axis=1)
    return rot_mat

def rot_mat_to_trace(r1, r2, sym):
    r2sym = tf.linalg.matmul(r1, sym)
    dg = tf.linalg.matmul(r2sym, r2)
    traces = tf.linalg.trace(dg)
    traces = tf.clip_by_value(traces, -1, 3)
    return traces
import numpy as np
from sklearn.decomposition import NMF

class NMFDataCompressor():
    def __init__(self, n_components):
        '''Instantiates the compressor model with set number of components'''
        self.compressor = NMF(n_components, max_iter=10_000)

    def fit(self, dataset, sample_size):
        '''
        - Extracts a random sample from the dataset 
        - Fits the compressor model
        '''
        # Get data array from the dataset
        data = dataset.get('data')
        
        # Extract a random sample
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        data_extract = data[idx[:sample_size]]
        
        # Fit the compressor
        self.compressor.fit(data_extract)

    def transform(self, data):
        '''Returns a compressed feature vector representation of the data'''
        return self.compressor.transform(data)
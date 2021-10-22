"""
@ Mallory Wittwer (mallory.wittwer@ntu.edu.sg)
@ Matteo Seita (mseita@ntu.edu.sg)

Demonstration of the method used to detect out-of-distribution examples using
Principal Components Analysis and a z-score model. Reproduces the z-score 
results shown in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class ZDetectorModel:
    def __init__(self, root, train_samples):
        self.scaler = StandardScaler()
        self.compressor = None
        self.root = root
        self.s0, self.s1 = 6, 72
        self.ds = 8
        self.train_samples = train_samples
        self.sample_meta = {}
        xtr, ytr = self.get_dataset(train_samples)
        self.training_data = xtr.reshape((len(xtr), self.s0 * self.s1))
        self.training_labels = ytr
        
        print(f'\nTraining data: {self.training_data.shape}')
        print(f'Training labels: {self.training_labels.shape}')
    
    
    def get_dataset(self, sample_names):
        """
        Loads training data.
        Args:
            sample_names : a list of sample names.
        """
        xtr = np.empty((0, self.s0, self.s1))
        ytr = np.empty((0, 3))
        for name in sample_names:
            path = os.path.join(self.root, f'training_sets/{name}.npy')
            dataset = np.load(path, allow_pickle=True).item()
            xtr_sample = dataset.get('xtr')
            ytr_sample = dataset.get('ytr')
            xtr = np.vstack((xtr, xtr_sample))
            ytr = np.vstack((ytr, ytr_sample))
        idx = np.arange(len(xtr))
        np.random.shuffle(idx)
        xtr = xtr[idx]
        ytr = ytr[idx]
        return xtr, ytr
    

    def compress(self, compressor):
        """
        Reduces dimensionality of the input using PCA.
        """
        self.compressor = compressor
        embedding = self.compressor.fit_transform(self.training_data)
        self.scaler.fit(embedding)
        print(f'\nFitted reduced data embedding! ({compressor.__class__.__name__})')
    

    def get_zscore(self, data):
        """
        Returns: the z-score indicator.
        """
        rde_scaled = self.scaler.transform(self.compressor.transform(data))
        zs = np.sqrt(np.sum(np.square(rde_scaled), axis=1))  # MSE of z-score across 2 dimensions
        return zs
    
    
    def evaluate_score(self, data, ds=8):
        """
        Args:
            data - A DRM dataset (4-D numpy array)
            ds (opt.) - image downscaling factor (higher = lower resolution)
        """
        zs_training = self.get_zscore(self.training_data)
        
        # IPF PLOT
        idx = np.arange(len(zs_training))
        np.random.shuffle(idx)
        zs_training = zs_training[idx]
        
        ### Specimen
        mini_data = data[::ds, ::ds]
        rx, ry, s0, s1 = mini_data.shape
        mini_data = mini_data.reshape((rx * ry, s0 * s1))
        zs = self.get_zscore(mini_data)
        
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
        ax.hist(zs, bins=20, color='#6dafd7', edgecolor='#222222', 
                linewidth=2, density=True)
        ax.set_xlabel('z')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 4)
        sns.distplot(zs_training[:5_000], ax=ax, hist=False,
                     kde_kws={'color': 'orange'})
        plt.show()
        
        # Show results
        zs = zs.reshape((rx, ry))
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        im = ax.imshow(zs, vmin=0, vmax=3.5)
        ax.axis('off')
        ax.figure.colorbar(im, ax=ax)
        plt.show()        


def run_test():
    """
    Test method.
    """
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(root, 'data/'))

    specimens = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    # Create anomaly detection model (specify specimens to use for training)
    model = ZDetectorModel(root, specimens)

    # Compress with PCA and 2 components
    model.compress(PCA(2))

    # Evaluate model on new specimen (downscaling is optional)
    new_specimen = np.load(root + '/drm_data.npy')
    model.evaluate_score(new_specimen, ds=8)
    
    new_specimen = np.load(root + '/anomaly_specimen.npy')
    model.evaluate_score(new_specimen, ds=8)
    

if __name__ == '__main__':
    """
    Expected result: plots of the z-score in two specimens containing anomalies.
    (i) Specimen with lack-of-fusion defects
    (ii) Specimen with underdeveloped surface structure
    => Should run in a few seconds on a modern laptop computer.
    """
    run_test()

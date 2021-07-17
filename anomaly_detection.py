"""
@ Mallory Wittwer (mallory.wittwer@ntu.edu.sg), May 2021
@ Matteo Seita (mseita@ntu.edu.sg)

Demonstration of the method used to detect out-of-distribution examples using
Principal Components Analysis and a z-score model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ZDetectorModel:
    def __init__(self, root, train_samples):
        self.scaler = StandardScaler()
        self.compressor = None
        self.root = root
        self.s0, self.s1 = 6, 72
        self.ds = 8
        self.train_samples = train_samples
        self.sample_meta = {}
        xtr = self.get_dataset(train_samples)
        self.training_data = xtr.reshape((len(xtr), self.s0 * self.s1))
        print('\nTraining data: ', self.training_data.shape)

    def get_dataset(self, sample_names):
        """
        Loads a DRM dataset from .npy file and creates training set (xtr)
        """
        xtr = np.empty((0, self.s0, self.s1))
        for name in sample_names:
            path = os.path.join(self.root, f'training_sets/{name}.npy')
            dataset = np.load(path, allow_pickle=True).item()
            xtr = np.vstack((xtr, dataset.get('xtr')))
        idx = np.arange(len(xtr))
        np.random.shuffle(idx)
        xtr = xtr[idx]
        return xtr

    def compress(self, compressor):
        """
        Reduces dimensionality of the input (for ex. using PCA)
        """
        self.compressor = compressor
        embedding = self.compressor.fit_transform(self.training_data)
        self.scaler.fit(embedding)
        print(f'\nFitted reduced data embedding! ({compressor.__class__.__name__})')

    def get_zscore(self, data):
        """
        Returns standard z-score
        """
        rde_scaled = self.scaler.transform(self.compressor.transform(data))
        zs = np.abs(rde_scaled)
        zs = np.sqrt(np.sum(np.square(zs), axis=1))
        return zs

    def evaluate_score(self, data, ds=8):
        """
        Evaluates model on data
        """
        mini_data = data[::ds, ::ds]
        rx, ry, s0, s1 = mini_data.shape
        mini_data = mini_data.reshape((rx * ry, s0 * s1))
        zs = self.get_zscore(mini_data).reshape((rx, ry))
        # Show results
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        im = ax.imshow(zs, vmin=0, vmax=3.5)
        ax.axis('off')
        ax.figure.colorbar(im, ax=ax)
        plt.show()


def run_test():
    """
    Reproduces the z score result shown in the paper.
    """
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(root, 'data/'))

    specimens = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    # Create anomaly detection model
    model = ZDetectorModel(root, specimens)  # specify specimens to use for training

    # Compress with PCA and 2 components
    model.compress(PCA(2))

    # Evaluate model on new specimen
    new_specimen = np.load(root + '/anomaly_specimen.npy')
    model.evaluate_score(new_specimen, ds=8)  # ds = downscaling factor (higher = lower resolution)


if __name__ == '__main__':
    '''
    Expected result: plot of z-score contrast on specimen that did not show
    good directional reflectance in the area close to the base plate.
    Should run in a few seconds on a modern laptop computer.
    '''
    run_test()

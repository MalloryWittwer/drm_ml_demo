"""
@ Mallory Wittwer (mallory.wittwer@ntu.edu.sg), June 2021
@ Matteo Seita (mseita@ntu.edu.sg)

Minimalistic code base to support the understanding of our
methodology and reproduce our results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from lib.ml_helpers import (
    CustomModel,
    voronoi_IPF_plot,
    visu_preds_and_targets,
)

import warnings

warnings.filterwarnings("ignore")


def define_model(checkpoint_path, train_samples, test_samples):
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(root, 'data/'))

    model = CustomModel(root,
                        bs=50, test_size=0.2, s0=6, s1=72,
                        checkpoint=checkpoint_path,
                        train_samples=train_samples,
                        test_samples=test_samples,
                        )

    model.create_model(
        kernelNeurons1=64,
        kernelNeurons2=64,
        denseNeurons1=256,
        denseNeurons2=128,
    )

    return model


def fit_model(model, lr=1e-3, patience=None, epochs=1, with_scheduler=False, save_it=True):
    model.set_learning_rate(model.run_lr_scheduler() if with_scheduler else lr)
    model.fit(patience=patience, epochs=epochs, save_it=save_it)
    return model


def run_test():
    """
    Reproduces the main results shown in the paper (should take a few minutes to run).
    """
    # Refer to all ten specimens
    specimens = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    # Initialize evaluation datasets
    df = pd.DataFrame(columns=['median', 'D_X', 'D_Y', 'D_Z'])
    moas_tot, eulers_tot = [], []

    # Iterate over specimens
    for split_index in range(len(specimens)):

        # Define split
        train_samples = np.delete(specimens, split_index)
        test_samples = [np.take(specimens, split_index)]

        # Define model
        model = define_model(
            checkpoint_path=f'../trained_models/{test_samples[0]}-100ep/',
            train_samples=train_samples,
            test_samples=test_samples,
        )

        # To train the model, at this point, we would use...
        # model = fit_model(model, 
        #                   lr=0.00031622776,
        #                   epochs=1,
        #                   with_scheduler=False,
        #                   save_it=False,
        #                   )

        # # Here instead, we reload the already trained model!
        train_path = os.path.join(os.path.realpath(__file__),
                                  f'../trained_models/{test_samples[0]}-100ep/')
        model.reload(train_path)

        # Evaluate the model
        moas, median_moa, me_x, me_y, me_z = model.evaluate()

        # Collect evaluation metrics
        df = df.append({
            'median': median_moa,
            'D_X': me_x,
            'D_Y': me_y,
            'D_Z': me_z,
        }, ignore_index=True)

        moas_tot.append(moas)
        eulers_tot.append(model.yte)

        if test_samples[0] == '08':  # Specimen shown in the paper

            ds = 1  # Downscaling factor (1 = native resolution)

            data = np.load(os.path.join(model.ROOT, 'drm_data.npy'))[::ds, ::ds]
            rx, ry, s0, s1 = data.shape
            data = data.reshape((rx * ry, s0, s1))
            dataset = tf.data.Dataset.from_tensor_slices(data).batch(50)
            eulers = np.load(os.path.join(model.ROOT, 'eulers.npy'))[::ds, ::ds]
            eulers = eulers.reshape((rx * ry, 3))

            # Get model prediction
            preds = model.predict(dataset)

            # Plot figure
            visu_preds_and_targets(preds, eulers, rx, ry)

    # Show total evaluation results
    print()
    print(df.describe().loc[['mean', 'std']])

    moas_tot = np.array([item for sublist in moas_tot for item in sublist])
    eulers_tot = np.array([item for sublist in eulers_tot for item in sublist])
    eulers_tot = eulers_tot.reshape((len(eulers_tot), 3))

    # Disorientation histogram
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    ax.hist(moas_tot, bins=20, color='#6dafd7', edgecolor='#222222', linewidth=2)
    ax.set_xlim(0, 62)
    ax.set_xlabel('Disorientation angle (deg)')
    ax.set_ylabel('Grain count')
    plt.show()

    # Error plotted in the IPF (X, Y, Z)
    voronoi_IPF_plot(eulers=eulers_tot, z_values=moas_tot, direction='x')
    voronoi_IPF_plot(eulers=eulers_tot, z_values=moas_tot, direction='y')
    voronoi_IPF_plot(eulers=eulers_tot, z_values=moas_tot, direction='z')


if __name__ == '__main__':
    '''
    Expected results: plot of IPF maps output from the EulerNet model compared
    to an EBSD reference, histogram of the disorienatation angle, IPF 
    distributions of the error, and printed table of evaluation metrics.
    '''
    run_test()

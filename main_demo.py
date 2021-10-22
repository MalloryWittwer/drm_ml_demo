"""
@ Mallory Wittwer (mallory.wittwer@ntu.edu.sg), June 2021
@ Matteo Seita (mseita@ntu.edu.sg)

Code base to support the understanding of our methodology and reproduce our 
main results.
"""

import os
import numpy as np
import tensorflow as tf

from lib.ml_helpers import (
    CustomModel,
    EvaluationManager,
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
        kernel_neurons_1=64,
        kenerl_neurons_2=64,
        dense_neurons_1=256,
        dense_neurons_2=128,
    )

    return model


def fit_model(model, lr=1e-3, patience=None, epochs=1, with_scheduler=False, save_it=True):
    model.set_learning_rate(model.run_lr_scheduler() if with_scheduler else lr)
    model.fit(patience=patience, epochs=epochs, save_it=save_it)
    return model


def generate_split(specimens):
    for split_index in range(len(specimens)):
        train_samples = np.delete(specimens, split_index)
        test_samples = [np.take(specimens, split_index)]
        yield train_samples, test_samples
        
        
def get_data_and_eulers(sample_name, ds=1):
    data_path = os.path.join(
        os.path.realpath(__file__),
        f'../data/samples/{sample_name}/drm_data.npy'
    )
    data = np.load(data_path)[::ds, ::ds]
    rx, ry, s0, s1 = data.shape
    data = data.reshape((rx * ry, s0, s1))
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(50)
    eulers_path = os.path.join(
        os.path.realpath(__file__),
        f'../data/samples/{sample_name}/eulers.npy'
    )
    eulers = np.load(eulers_path)[::ds, ::ds].reshape((rx * ry, 3))
    
    return dataset, eulers, rx, ry


def run_test():
    """
    Reproduces the main results shown in the paper.
    Should take a few minutes to run on a modern laptop computer.
    """
    # Initialize evaluation datasets
    # (i) On a grain level
    eval_grain_level = EvaluationManager()
    
    # # (ii) On a pixel level
    # # /!\ Data only available upon request! (~12 Gb)
    # # Please contact the corresponding author to test that functionality.
    # eval_pixel_level = EvaluationManager()

    # Iterate over specimens (cross-validation)
    specimens = ['{:02d}'.format(k) for k in range(1,11)] # specimen names
    for train_samples, test_samples in generate_split(specimens):

        # Define the model
        model = define_model(
            checkpoint_path=f'../trained_models/{test_samples[0]}-100ep/',
            train_samples=train_samples,
            test_samples=test_samples,
        )

        # # To train the model, at this point, we would use...
        # model = fit_model(model, 
        #                   lr=0.00031622776,
        #                   epochs=1,
        #                   with_scheduler=False,
        #                   save_it=False,
        #                   )

        # Here instead, we reload the already trained model!
        train_path = os.path.join(
            os.path.realpath(__file__),
            f'../trained_models/{test_samples[0]}-100ep/'
        )
        model.reload(train_path)
        
        # Evaluate the model on a grain level   
        eval_grain_level.evaluate(
            preds=model.predict(model.testing_dataset), 
            eulers=model.yte
        )
        
        if test_samples[0] == '08':  # Specimen shown in the paper            
            dataset, eulers, rx, ry = get_data_and_eulers(test_samples[0], ds=4)
            preds = model.predict(dataset)
            visu_preds_and_targets(preds, eulers, rx, ry)
        
        # # Un-comment for evaluation on a pixel level (requires all 12 Gb of data)
        # dataset, eulers, rx, ry = get_data_and_eulers(test_samples[0], ds=8)        
        # eval_pixel_level.evaluate(
        #     preds=model.predict(dataset), 
        #     eulers=eulers
        # )
        
    eval_grain_level.display("Evaluation on a grain level:")
    eval_grain_level.plot_histogram("Grain count")
    eval_grain_level.plot_IPFs()
    
    # # Un-comment for evaluation on a pixel level (requires all 12 Gb of data)
    # eval_pixel_level.display("Evaluation on a pixel level:")
    # eval_pixel_level.plot_histogram("Pixel count")


if __name__ == '__main__':
    '''
    Expected results: plot of IPF maps output from the EulerNet model compared
    to an EBSD reference, histogram of the disorienatation angle, IPF 
    distributions of the error, and printed table of evaluation metrics.
    '''
    run_test()

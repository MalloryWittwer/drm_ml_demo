# EulerNet: A Machine Learning Approach to Map Crystal Orientation by Optical Microscopy

This repository contains the Python code necessary to reproduce the results presented in our publication.

### Installation

##### Code

Please clone this repository to your local machine.

##### Dependencies

We recommend executing the code in the provided virtual environment `environment.yml` using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (to install Anaconda: see [here](https://www.anaconda.com/)). To create the environement, use:

`conda env create -f environment.yml`

This will create an environment with the name *drm_ml*. To activate that environment, use:

`conda activate drm_ml`

We tested the code using the following dependencies:

- python 3.6.13
- numpy 1.19.2
- pandas 1.1.5
- matplotlib 3.3.4
- scikit-image 0.17.2
- scikit-learn 0.24.2
- tensorflow 2.1.0

##### Data
Please download the **data folder** (3.4 Gb) from [here](DOI: 10.17632/z8bh7n5b7d.1) (will open a Mendeley Dataset) and copy it next to the root directory of the repository:

| drm_ml_demo
    | **data**
    | lib
    | trained models
    | tests

### Description

The **data** folder contains (i) all training and evaluation sets used to derive the results presented in our publication and (ii) three additional files: 
- **drm_data.npy** : A 4D numerical matrix (shape (x, y, theta, phi), type uint8) representing the experimental DRM dataset of the test specimen showcased in Figure 3 of the paper.
- **eulers.npy** : The corresponding matrix of Euler angles measured by EBSD for this test specimen (shape (x, y, 3), type float32).
- **anomaly_specimen.npy** : The DRM dataset of the specimen shown in Figure 6 of the paper to demonstrate the detection of out-of-distribution data.

The **lib** folder contains the Python code to process the data and implement and test our machine learning models to reproduce our results.

The **trained_models** folder contains 10 EulerNet models trained independently on the different cross-validation splits.

### Steps to reproduce our results

The repository root contains three test files:

- Execute **main_demo.py** to reproduce the machine learning prediction and evaluate performance.
- Execute **anomaly_detection.py** to reproduce our anomaly detection results.
- Execute **data_extraction.py** for a demonstration of the process used to select data for training and test sets.

### Inquiries
For any inquiry, please contact the corresponding author.

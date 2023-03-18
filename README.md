# ResNet-Model_with_ActiveSampling-Strategy
In this project, a new approach is presented to accelerate the training of neural networks.

# Active Learning with Similarity Measures
This is a Python project that implements an active learning approach using similarity measures. The project includes code for selecting a subset of representative images from a large dataset in order to train a machine learning model more efficiently. The package is particularly useful for datasets with a large number of classes, where it is difficult to find representative images.

The package consists of six code snippets:

main.py: This is the main script that runs the active learning process. It takes in a dataset of images and labels and uses similarity measures to select a subset of representative images from each class. It then returns the selected images and their corresponding labels for use in training a machine learning model.

active_sampling.py: This script contains the Similarity_measures class, which calculates the similarity between two images using the Peak Signal to Noise Ratio (PSNR). It also includes a function for calculating the PSNR between two images.

ResNetModel.py: This script defines the machine learning model that will be trained using the selected images.

Dataloader.py: This script contains the code for loading the Data.

load_data.py: This script contains the code for actually loading the Data that we want for the training.

preprocessing.py: This script contains the preprocess class, which sorts the images by class and selects a subset of representative images from each class.

# Installation
To install the package, clone this repository and run the following command:

pip install -r requirements.txt
This will install all the necessary packages for running the active learning process.

# Usage
To use the package, run the following command:

css
python main.py --data_path /path/to/dataset --num_images 100
This will select 100 representative images from each class in the dataset and return their corresponding labels. The selected images and labels can then be used to train a machine learning model.

# Contributing
Contributions to this project are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

<div align="center">    
 
# Few-Zero shot learning using CLIP

</div>
 
## Description   
This is a simple few shot classifier working on top of features extracted from [CLIP architecture](https://github.com/openai/CLIP). Given training, test and validation set test set is the best performing model. On the other side, we have tested ViT-B/16 and ViT-B/32 CLIP models. The latter provided the best results.

## Provided dataset

The dataset ”sleevetypes” provided is specifically structured for few-shot learning
experiments. It is divided into three primary directories named: train,
test, and val. Inside each of these directories, there are subfolders that represent
different sleeve types. These subfolder names directly correspond to the
sleeve types, such as ”balloon” or ”raglan”. Each of these subfolders contains
image samples that depict its respective sleeve type. Thus, you can infer the
sleeve type directly from the name of the subfolder in the train, test, and val
directories.

## Installation

- PyTorch: pip install torch.
- torchvision: pip install torchvision.
- PIL (Python Imaging Library): pip install pillow.
- OS: import os
- random: import random
- matplotlib: import matplotlib.pyplot

## Usage

- Import the necessary libraries and set up the device (CPU or GPU) for processing.
- Load the CLIP model and define image transformations.
- Prepare your data in the required format using datasets.ImageFolder. Ensure that your dataset is organized into train, test, and validation directories.
- Create data loaders for your datasets.
- Define functions to extract prototypes and classify images based on cosine similarity.
- Evaluate the model on the validation dataset and calculate the accuracy.
- Test the model on the test dataset and calculate the accuracy.

## How to run

### run module

python main.py

## Some results

### Validation and test accuracies

Validation = 50%
Test = 67.74%

```

```

## Summary of the code and approach:

### Model Loading:

- The CLIP model ("ViT-B/32") is loaded with a specified device (GPU if available).

#### Dataset Preparation:

- The code assumes the existence of training, testing, and validation directories containing subfolders for different sleeve types, each with image samples.

#### Image transformations suitable for CLIP are defined.

- Datasets and DataLoaders for training, testing, and validation are created using PyTorch's ImageFolder and DataLoader.

#### Prototypes Extraction:

- A function extract_prototypes is defined to extract class prototypes from the training dataset. These prototypes are calculated as the mean of embeddings of a few support images per class.
- The prototypes are stored in a dictionary, where each class label corresponds to its prototype.

#### Image Classification:

- A function classify_image is defined to classify a single image using the prototypes. It calculates the cosine similarity between the image's embedding and each class prototype and selects the class with the highest similarity as the predicted label.

#### Validation and Testing:

- The classifier is evaluated on the validation and test datasets, and accuracy is calculated.
- The code prints the validation and test accuracy.

#### Results:

- The code showcases the few-shot learning capability of CLIP, achieving classification accuracy on the "sleevetypes" dataset.
- The achieved validation and test accuracy results are printed.

## citation

@article{radford2021learning,
title={Learning Transferable Visual Models From Natural Language Supervision},
author={Radford, Alec and Narasimhan, Karthik and Rockt{\"a}schel, Tim and et al.},
journal={arXiv preprint arXiv:2103.00020},
year={2021}
}

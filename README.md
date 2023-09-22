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

-Import the necessary libraries and set up the device (CPU or GPU) for processing.
-Load the CLIP model and define image transformations.
-Prepare your data in the required format using datasets.ImageFolder. Ensure that your dataset is organized into train, test, and validation directories.
-Create data loaders for your datasets.
-Define functions to extract prototypes and classify images based on cosine similarity.
-Evaluate the model on the validation dataset and calculate the accuracy.
-Test the model on the test dataset and calculate the accuracy.

## How to run

### run module

python main.py

## Some results

### Validation and test accuracies

Validation = 50%
Test = 67.74%

```

```

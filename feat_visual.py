import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import models
import data

import os

#define the data we uploaded as evaluation data and apply the transformations
evalset = torchvision.datasets.ImageFolder(root="../Training Data/Places_Nature10")

data_dir = '../Training Data/Places_Nature10/val'

training_dataset = DataLoader(
    data.Places365(path_to_index_file='../Training Data/Places_Nature10', index_file_name='val.txt'),
    batch_size=16, shuffle=True, drop_last=True,
    collate_fn=data.image_label_list_of_masks_collate_function)

dataset_sizes = len(training_dataset)

print("Loaded {} images under val".format(dataset_sizes))

# print("Classes: ")
# print(training_dataset.classes)

modelVGG = models.VGG16(path_to_pre_trained_model="./pre_trained_models/VGG16_P10_50ep.pt")
print(modelVGG)

inputs, classes, masks = next(iter(training_dataset))
modelVGG.eval()
modelVGG.to('cuda')
inputs = inputs.to('cuda')

features = modelVGG(inputs)



for layer in features:
    layer_vis = layer[0].shape
    print(layer_vis)

plt.figure(figsize=(50, 10))
layer_visual = features[0][0]
for i, filter in enumerate(layer_visual):
    if i == 64:
        break
    filter = filter.cpu()
    filter = filter.detach().numpy()
    plt.subplot(8, 8, i + 1)
    plt.imshow(filter)
    plt.axis("off")

plt.show()

layer_visual = features[1][0]
for i, filter in enumerate(layer_visual):
    if i == 16:
        break
    filter = filter.cpu()
    filter = filter.detach().numpy()
    plt.subplot(2, 8, i + 1)
    plt.imshow(filter)
    plt.axis("off")

plt.show()

layer_visual = features[2][0]
for i, filter in enumerate(layer_visual):
    if i == 16:
        break
    filter = filter.cpu()
    filter = filter.detach().numpy()
    plt.subplot(2, 8, i + 1)
    plt.imshow(filter)
    plt.axis("off")

plt.show()

layer_visual = features[3][0]
for i, filter in enumerate(layer_visual):
    if i == 16:
        break
    filter = filter.cpu()
    filter = filter.detach().numpy()
    plt.subplot(2, 8, i + 1)
    plt.imshow(filter)
    plt.axis("off")

plt.show()

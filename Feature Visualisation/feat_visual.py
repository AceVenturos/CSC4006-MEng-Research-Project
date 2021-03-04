import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os

#define the data we uploaded as evaluation data and apply the transformations
evalset = torchvision.datasets.ImageFolder(root="../../Training Data/Places_Nature10")

data_dir = '../../Training Data/Places_Nature10/val'

# Images to Tensors
tf = transforms.Compose([transforms.ToTensor()])


image_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=tf)

torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True, num_workers=0)

dataset_sizes = len(image_dataset)

print("Loaded {} images under val".format(dataset_sizes))

print("Classes: ")
class_names = image_dataset.classes
print(image_dataset.classes)

dataiter = iter(image_dataset)
images, labels = dataiter.__next__()

print(labels)

#shape of images bunch
print(images.shape)
#shape of single image in a bunch
print(images[0].shape)

#label of the image
print(labels[0])

model =

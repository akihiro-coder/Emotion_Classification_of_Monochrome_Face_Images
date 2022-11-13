import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import cv2 as cv
import os
import pandas as pd
import glob
import random


# set machine resource
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])


# prepare dataset
train_dataset = datasets.ImageFolder('./data/train/label', transform=transform)

# prepare dataloader
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# label name
emotion_name = ('sad', 'angry', 'neutral', 'happy')

# 画像の確認(get images and labels from dataloader)
#data_iter = iter(train_dataloader)
#imgs, labels = data_iter.next()
#img = imgs[0]
#label = labels[0]
#print(label)
#save_image(img, 'first_img.png')


# define model structure
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init()

        

# define criterion


# define optimizer


# train loop


# validation loop


# graph result

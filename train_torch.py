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



# prepare dataset



# prepare dataloader 



# label name



# 画像の確認(get images and labels from dataloader)





# define model structure



# define criterion


# define optimizer



# train loop 


# validation loop 




# graph result






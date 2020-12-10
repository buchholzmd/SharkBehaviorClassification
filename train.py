import os
import yaml
import pywt
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from utils import *
from datasets.SharkBehaviorDataset import SharkBehaviorDataset

########################### model params ###########################
with open('./configs/config.yml', 'r') as stream:
    config = yaml.safe_load(stream)

expDim     = config['DATA_DIM']
wavelet    = config['WAVELET']
archType   = config['ARCH_TYPE']
testSplit  = config['TEST_SPLIT']
modelType  = config['MODEL_TYPE']
batch_size = config['BATCH_SIZE']

folder_, results_folder, results_file = get_exp_paths(config)
    
########################### save/load ###########################

data_folder = os.path.join('./datasets/data', expDim)

train_X, train_Y = load_data(data_folder, 'train')
val_X, val_Y = load_data(data_folder, 'val')

########################### print folder ###########################

print(data_folder)

########################### expand dims ###########################

cnn = (modelType == 'cnn') or (modelType == 'rcnn')

if cnn: 
    train_X = np.transpose(train_X, (0, 2, 1))
    val_X   = np.transpose(val_X, (0, 2, 1))
    
    if expDim == '2d':
        train_X = np.expand_dims(train_X, axis=1)
        val_X   = np.expand_dims(val_X, axis=1)

########################### load up dataset ###########################

train_dataset = SharkBehaviorDataset(train_X, labels=train_Y, train=True)
val_dataset   = SharkBehaviorDataset(val_X, labels=val_Y, train=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

model, optimizer, sched = get_model(config)
criterion = nn.CrossEntropyLoss()

# print(summary(model, (50,1)))

########################### train ###########################

train(model, train_loader, val_loader, {}, optimizer, sched, folder_, rnn=(not cnn))

########################### disp loss/acc ###########################

# train_acc_series = pd.Series(mean_train_acc)
# val_acc_series = pd.Series(mean_val_acc)
# train_acc_series.plot(label="train")
# val_acc_series.plot(label="validation")
# plt.legend()

# train_acc_series = pd.Series(mean_train_losses)
# val_acc_series = pd.Series(mean_val_losses)
# train_acc_series.plot(label="train")
# val_acc_series.plot(label="validation")
# plt.legend()
 
import os
import yaml
import pywt
import torch
import gpytorch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns

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
    
# DATA
expDim     = config['DATA_DIM']
wavelet    = config['WAVELET']
archType   = config['ARCH_TYPE']
testSplit  = config['TEST_SPLIT']
modelType  = config['MODEL_TYPE']
batch_size = config['BATCH_SIZE']

paths_dict = get_exp_paths(config)

folder_ = paths_dict['folder']
results_folder = paths_dict['results_dir']
results_file = paths_dict['results_fn']
split_path = paths_dict['split_path']
    
########################### save/load ###########################

train_X, train_Y = load_data(split_path, 'train')
val_X, val_Y = load_data(split_path, 'val')

########################### print folder ###########################

print(split_path)

########################### expand dims ###########################

cnn = (modelType == 'cnn') or (modelType == 'rcnn')

if cnn:
    if expDim == '2d':
        train_X = np.expand_dims(train_X, axis=1)
        val_X   = np.expand_dims(val_X, axis=1)
    else:
        train_X = np.transpose(train_X, (0,2,1))
        val_X   = np.transpose(val_X, (0,2,1))

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

if modelType == 'svdkl':
    criterion = gpytorch.mlls.PredictiveLogLikelihood(model['likelihood'], model['model'].gp_layer, num_data=len(train_loader.dataset))

    kernel_learning = True

else:
    criterion = nn.CrossEntropyLoss()

    if cnn:
        if expDim == '2d':
            print(summary(model, (1, 50, 6)))
        else:
            print(summary(model, (1, 50)))

    kernel_learning = False

test_path   = config['TEST_PATH']
model_path = os.path.join(folder_,test_path)
print("Loading model from path: ", model_path)

train(config, model, criterion, train_loader, val_loader, {}, optimizer, sched, folder_, rnn=(not cnn), kernel_learning=kernel_learning, model_path=model_path)


 
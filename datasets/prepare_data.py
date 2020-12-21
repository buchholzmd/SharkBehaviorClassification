import os
import argparse
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

from utils import *

from torch.autograd import Variable

import seaborn as sns

import h5py
#import pywt

import scipy

with open('../configs/config.yml', 'r') as stream:
    config = yaml.safe_load(stream)

expDim     = config['DATA_DIM']
num_train  = config['NUM_TRAIN']
num_val    = config['NUM_VAL']
num_test   = config['NUM_TEST']
split_type = config['SPLIT']
split_path = config['SPLIT_PATH']

df_list = load_data(config)

if expDim == '1d':
    features = ['ODBA']
    dims = 1

elif expDim == '2d':
    features = ['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']
    dims = 6
    
else:
    print("Not a valid data dimension")
    
df_dict = split_data(df_list, config)
    
df_dict['train'] = df_dict['train'][features + ['Label']]
df_dict['val'] = df_dict['val'][features + ['Label']]
df_dict['test'] = df_dict['test'][features + ['Label']]    

df_dict = normalize(df_dict, features)

df_dict['train'] = group_times(df_dict['train'])
df_dict['val'] = group_times(df_dict['val'])
df_dict['test'] = group_times(df_dict['test'])

print('Sampling training sequences')
X_train, Y_train = sample_sequences(df_dict['train'], features, num_samples=num_train, dims=dims)
X_train, Y_train = shuffle(X_train, Y_train, random_state=33)
print()

print('Sampling validation sequences')
X_val, Y_val = sample_sequences(df_dict['val'], features, num_samples=num_val, dims=dims)
X_val, Y_val = shuffle(X_val, Y_val, random_state=33)
print()

print('Sampling testing sequences')
if split_type == 'experiment': 
    X_test, Y_test = sample_sequences(df_dict['test'], features, num_samples=num_test, dims=dims)
elif split_type == 'full':
    X_test, Y_test = sample_sequences(df_dict['test'], features, num_samples=num_test, dims=dims)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=33)
    
print("="*44)
print(X_train.shape)
print(Y_train.shape)
print()

print(X_val.shape)
print(Y_val.shape)
print()

print(X_test.shape)
print(Y_test.shape)
print()

split_path = '.' + split_path
split_path = os.path.join(split_path, split_type + '_split')

data_path = os.path.join(split_path, expDim)

if not os.path.exists(data_path):
    os.makedirs(data_path)
        
train_path = os.path.join(data_path, 'train')  

if not os.path.exists(train_path):
    os.makedirs(train_path)

print("Saving at: " + train_path)
write(X_train, Y_train, os.path.join(train_path, 'data.hdf5'))
        
val_path = os.path.join(data_path, 'val')  

if not os.path.exists(val_path):
    os.makedirs(val_path)

print("Saving at: " + val_path)
write(X_val, Y_val, os.path.join(val_path, 'data.hdf5'))
        
test_path = os.path.join(data_path, 'orig/test')  

if not os.path.exists(test_path):
    os.makedirs(test_path)

print("Saving at: " + test_path)
write(X_test, Y_test, os.path.join(test_path, 'data.hdf5'))

print("Saving successful :) ... ")